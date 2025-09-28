"""
BeyondMimic ONNX Policy for MuJoCo Deployment

This policy class loads and uses the exported ONNX model from whole_body_tracking
for deployment in MuJoCo with Vuer.
"""

import numpy as np
import onnx
import onnxruntime as ort
from typing import List, Optional, Dict, Any
import xml.etree.ElementTree as ET
import os
import sys
import yaml
import re
from pathlib import Path

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from vuer_sim_example.params import Robot, Observation, Policy

def get_mj_actuator_joint_names_recursive(xml_path: str) -> List[str]:
    """Extract joint names from MuJoCo XML file (same as in vuerdeployment.py)"""
    import xml.etree.ElementTree as ET
    import os

    xml_path = os.path.abspath(xml_path)
    seen = set()
    out = []

    def _walk(path):
        path = os.path.abspath(path)
        if path in seen or not os.path.exists(path):
            return
        seen.add(path)

        tree = ET.parse(path)
        root = tree.getroot()
        base_dir = os.path.dirname(path)

        # collect actuator → joint names in the order they appear
        for elem in root.findall(".//actuator/*"):
            j = elem.get("joint")
            if j:
                out.append(j)

        # follow nested includes
        for inc in root.findall(".//include"):
            inc_file = inc.get("file")
            if not inc_file:
                continue
            inc_path = inc_file if os.path.isabs(inc_file) else os.path.join(base_dir, inc_file)
            _walk(inc_path)

    _walk(xml_path)
    return out

def load_env_yaml(yaml_path: str = None) -> Dict[str, Any]:
    """Load and parse the env.yaml configuration file"""
    if yaml_path is None:
        # Default to relative path from this file
        current_dir = Path(__file__).parent
        yaml_path = current_dir.parent / "vuer_sim_example" / "envs" / "env.yaml"
    
    with open(yaml_path, 'r') as f:
        # Use unsafe_load to handle Python-specific tags like !!python/tuple
        config = yaml.unsafe_load(f)
    return config

def expand_joint_config(config_dict: Dict[str, float], joint_names: List[str]) -> Dict[str, float]:
    """Expand regex patterns in config to actual joint names"""
    result = {}
    for pattern, value in config_dict.items():
        # Convert .* patterns to regex
        pattern_re = pattern.replace('.*', '.*')
        regex = re.compile(f"^{pattern_re}$")
        
        for joint_name in joint_names:
            if regex.match(joint_name):
                # Only set if not already set (first match wins)
                if joint_name not in result:
                    result[joint_name] = value
    return result

class BeyondMimicPolicy:
    """
    Policy class for BeyondMimic exported ONNX models.

    This class loads the exported ONNX model from whole_body_tracking and provides
    the same interface as the BasePolicy in vuer-sim-example for easy integration.
    """

    def __init__(self, onnx_path: str, motion_npz_path: str, mj_joint_names: Optional[List[str]] = None, 
                 env_yaml_path: str = None):
        """Initialize BeyondMimic policy with external motion data
        
        Args:
            onnx_path: Path to ONNX model 
            motion_npz_path: Path to NPZ file with reference motion data
            mj_joint_names: Optional list of MuJoCo joint names for mapping
        """
        # Load ONNX model
        self.onnx_session = ort.InferenceSession(onnx_path)
        self.input_names = [inp.name for inp in self.onnx_session.get_inputs()]
        self.output_names = [out.name for out in self.onnx_session.get_outputs()]
        # Load motion data from NPZ
        motion_data = np.load(motion_npz_path)
        self.motion_joint_pos = motion_data['joint_pos']  # Shape: (frames, joints)
        self.motion_joint_vel = motion_data['joint_vel']  # Shape: (frames, joints)
        self.motion_fps = int(motion_data['fps'])
        self.motion_frames = self.motion_joint_pos.shape[0]
        
        # Verify this is the new format
        assert "obs" in self.input_names, f"Expected 'obs' input, got {self.input_names}"
        assert "time_step" in self.input_names, f"Expected 'time_step' input, got {self.input_names}"
        assert "actions" in self.output_names, f"Expected 'actions' output, got {self.output_names}"

        # Load model metadata
        model = onnx.load(onnx_path)
        self.metadata = {p.key: p.value for p in model.metadata_props}

        # Load configuration from env.yaml
        env_config = load_env_yaml(env_yaml_path)
        
        # Load metadata from ONNX model
        self.joint_names = self._get_metadata("joint_names", "").split(",") if self._get_metadata("joint_names") else []
        
        # Get default joint positions from env.yaml
        joint_pos_config = env_config['scene']['robot']['init_state']['joint_pos']
        training_defaults = expand_joint_config(joint_pos_config, self.joint_names)
        
        # Fill in any missing joints with 0
        for joint_name in self.joint_names:
            if joint_name not in training_defaults:
                training_defaults[joint_name] = 0.0
        
        # Map defaults to ONNX joint order
        self.default_joint_pos = np.array([training_defaults.get(name, 0.0) for name in self.joint_names], dtype=np.float32)
        
        # Set number of joints
        self.n = len(self.joint_names)
        
        # Get observation dimension from model
        obs_input = next((inp for inp in self.onnx_session.get_inputs() if inp.name == "obs"), None)
        self.obs_dim = obs_input.shape[1] if obs_input else 160

        # Load action scales from env.yaml
        action_scale_config = env_config['actions']['joint_pos']['scale']
        self.training_action_scales = expand_joint_config(action_scale_config, self.joint_names)
        
        # Load PD gains from env.yaml actuators section
        self.pd_gains = {'kp': {}, 'kd': {}}
        actuator_configs = env_config['scene']['robot']['actuators']
        
        for actuator_group in actuator_configs.values():
            if 'stiffness' in actuator_group:
                stiffness = actuator_group['stiffness']
                if isinstance(stiffness, dict):
                    kp_expanded = expand_joint_config(stiffness, self.joint_names)
                    self.pd_gains['kp'].update(kp_expanded)
                else:
                    # Single value for all joints in this group
                    if 'joint_names_expr' in actuator_group:
                        for pattern in actuator_group['joint_names_expr']:
                            pattern_re = pattern.replace('.*', '.*')
                            regex = re.compile(f"^{pattern_re}$")
                            for joint_name in self.joint_names:
                                if regex.match(joint_name) and joint_name not in self.pd_gains['kp']:
                                    self.pd_gains['kp'][joint_name] = stiffness
                                    
            if 'damping' in actuator_group:
                damping = actuator_group['damping']
                if isinstance(damping, dict):
                    kd_expanded = expand_joint_config(damping, self.joint_names)
                    self.pd_gains['kd'].update(kd_expanded)
                else:
                    # Single value for all joints in this group
                    if 'joint_names_expr' in actuator_group:
                        for pattern in actuator_group['joint_names_expr']:
                            pattern_re = pattern.replace('.*', '.*')
                            regex = re.compile(f"^{pattern_re}$")
                            for joint_name in self.joint_names:
                                if regex.match(joint_name) and joint_name not in self.pd_gains['kd']:
                                    self.pd_gains['kd'][joint_name] = damping

        # CRITICAL: Create correct mapping from ONNX to MuJoCo
        # MuJoCo order from XML (lines 531-559):
        expected_mj_order = [
            'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
            'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
            'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
            'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
            'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
            'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
            'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
            'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
        ]
        
        # Create direct mapping from ONNX order to expected MuJoCo order
        self.onnx_to_mj = np.zeros(self.n, dtype=np.int32)
        for i, onnx_name in enumerate(self.joint_names):
            if onnx_name in expected_mj_order:
                self.onnx_to_mj[i] = expected_mj_order.index(onnx_name)
            else:
                self.onnx_to_mj[i] = -1
        # Motion timing and state - motion data is embedded in ONNX
        self.time_step = 0
        self.prev_action = np.zeros(self.n, dtype=np.float32)

    def _get_metadata(self, key: str, default: str = "") -> str:
        """Get metadata value by key"""
        return self.metadata.get(key, default)
    
    def _csv_to_float_array(self, s: str) -> np.ndarray:
        """Parse comma-separated string to float array"""
        if not s:
            return np.array([])
        return np.array([float(x) for x in s.split(",") if x.strip()], dtype=np.float32)

    def reset(self):
        """Reset policy state"""
        self.prev_action.fill(0.0)
        self.time_step = 0
    
    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate vector by inverse quaternion (from base_policy)"""
        q_w = q[0]
        q_vec = q[1:]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0
        dot_product = np.sum(q_vec * v)
        c = q_vec * dot_product * 2.0
        return a - b + c

    def build_observation(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        """Build 160-dim observation"""
        # CRITICAL: Use model's OUTPUT as motion reference (like C++ implementation)
        # For first timestep, use defaults or NPZ data
        if not hasattr(self, 'last_joint_pos_output'):
            # Initialize with NPZ data for first timestep
            motion_frame = 0
            self.last_joint_pos_output = self.motion_joint_pos[motion_frame].astype(np.float32)
            self.last_joint_vel_output = self.motion_joint_vel[motion_frame].astype(np.float32)
        
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        offset = 0
        
        # Command: Use MODEL'S PREVIOUS OUTPUT as reference (58 dims)
        motion_input = np.concatenate([self.last_joint_pos_output, self.last_joint_vel_output], axis=0)
        obs[offset:offset + 58] = motion_input
        offset += 58
        
        # Motion anchor position: Use base position relative to reference - 3 dims
        base_pos = qpos[:3]
        obs[offset:offset + 3] = base_pos  # Real base position
        offset += 3
        
        # Motion anchor orientation: Use projected gravity + quaternion info - 6 dims
        base_quat = qpos[3:7]  # [w, x, y, z]
        # Calculate projected gravity (crucial for balance like in base_policy)
        gravity_vec = np.array([0, 0, -1])
        projected_gravity = self.quat_rotate_inverse(base_quat, gravity_vec)
        obs[offset:offset + 3] = projected_gravity
        obs[offset + 3:offset + 6] = base_quat[:3]  # Use first 3 quaternion components
        offset += 6
        
        # Base linear velocity - 3 dims
        obs[offset:offset + 3] = qvel[:3]
        offset += 3
        
        # Base angular velocity - 3 dims  
        obs[offset:offset + 3] = qvel[3:6]
        offset += 3
        
        # Joint positions (relative to default) - 29 dims
        current_joint_pos = qpos[7:7+self.n]
        obs[offset:offset + self.n] = current_joint_pos - self.default_joint_pos
        offset += self.n
        
        # Joint velocities - 29 dims
        current_joint_vel = qvel[6:6+self.n]
        obs[offset:offset + self.n] = current_joint_vel
        offset += self.n
        
        # Previous actions - 29 dims (final part of YOUR 160-dim format)
        obs[offset:offset + self.n] = self.prev_action
        
        # Total should be exactly 160: 58 + 3 + 6 + 3 + 3 + 29 + 29 + 29 = 160
        return obs.reshape(1, -1)

    def predict(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        """Predict action from current state"""
        obs = self.build_observation(qpos, qvel)
        
        # ONNX inference
        input_dict = {"obs": obs, "time_step": np.array([[self.time_step]], dtype=np.float32)}
        output_list = self.onnx_session.run(self.output_names, input_dict)
        action = output_list[0].astype(np.float32).ravel()
        
        # CRITICAL: Save model's motion reference outputs for next timestep
        # This is what the C++ implementation does!
        if 'joint_pos' in self.output_names and 'joint_vel' in self.output_names:
            joint_pos_idx = self.output_names.index('joint_pos')
            joint_vel_idx = self.output_names.index('joint_vel')
            self.last_joint_pos_output = output_list[joint_pos_idx].ravel()
            self.last_joint_vel_output = output_list[joint_vel_idx].ravel()
        
        # Apply training action scaling
        action_scales = np.array([self.training_action_scales.get(name, 0.25) for name in self.joint_names], dtype=np.float32)
        scaled_action = action * action_scales
        
        # CRITICAL INSIGHT: The model outputs joint_pos which IS the target!
        # The action is just the control signal, but joint_pos is what we should track
        USE_MODEL_JOINT_POS = True
        
        if USE_MODEL_JOINT_POS and 'joint_pos' in self.output_names:
            # Use the model's joint_pos output directly as target
            target_joint_pos = self.last_joint_pos_output
        else:
            # Fallback to using action + default
            target_joint_pos = scaled_action + self.default_joint_pos
        
        # Direct mapping from ONNX order to MuJoCo order
        if self.onnx_to_mj is not None:
            target_mj = np.zeros(29, dtype=np.float32)  # Always 29 joints for G1
            for i in range(self.n):
                mj_idx = self.onnx_to_mj[i]
                if mj_idx != -1:
                    target_mj[mj_idx] = target_joint_pos[i]
        else:
            target_mj = target_joint_pos
        
        # Update state
        self.time_step += 1
        self.prev_action = action.copy()
        
        # PD control using gains from env.yaml
        position_error = target_mj - qpos[7:]
        velocity_error = -qvel[6:]
        
        # Use per-joint gains from env.yaml
        kp_array = np.zeros(29, dtype=np.float32)
        kd_array = np.zeros(29, dtype=np.float32)
        
        for i in range(self.n):
            mj_idx = self.onnx_to_mj[i]
            if mj_idx != -1:
                joint_name = self.joint_names[i]
                # Use gains from env.yaml or fallback to reasonable defaults
                kp_array[mj_idx] = self.pd_gains['kp'].get(joint_name, 100.0)
                kd_array[mj_idx] = self.pd_gains['kd'].get(joint_name, 10.0)
        
        torques = kp_array * position_error + kd_array * velocity_error
        
        # Clip to reasonable motor limits
        torques = np.clip(torques, -150.0, 150.0)
        
        return torques

    def get_info(self) -> dict:
        """Get policy information"""
        return {
            "n_joints": self.n,
            "obs_dim": self.obs_dim,
            "joint_names": self.joint_names,
            "time_step": self.time_step,
        }

# Example usage function
def create_beyond_mimic_policy(onnx_path: str, motion_npz_path: str, xml_path: str = None, 
                               env_yaml_path: str = None) -> BeyondMimicPolicy:
    """Factory function to create a BeyondMimic policy with joint name mapping
    
    Args:
        onnx_path: Path to ONNX model
        motion_npz_path: Path to NPZ file with reference motion data
        xml_path: Path to MuJoCo XML (optional, for joint name mapping)
        env_yaml_path: Path to env.yaml configuration file
    """

    # Create policy with env config
    policy = BeyondMimicPolicy(onnx_path, motion_npz_path, env_yaml_path=env_yaml_path)

    return policy