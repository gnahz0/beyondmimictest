import numpy as np
import onnxruntime
from ..params import Robot, Observation, Policy


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector by inverse of quaternion using efficient formula.
    
    Args:
        q: quaternion array of shape (N, 4) - [w, x, y, z]
        v: vector array of shape (N, 3) - [x, y, z]
    
    Returns:
        rotated vector of shape (N, 3)
    """
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0)[:, np.newaxis]
    b = np.cross(q_vec, v) * q_w[:, np.newaxis] * 2.0
    dot_product = np.sum(q_vec * v, axis=1, keepdims=True)
    c = q_vec * dot_product * 2.0
    return a - b + c


def quat_rotate_inverse_numpy(q, v):
    """Alias for quat_rotate_inverse for backward compatibility"""
    return quat_rotate_inverse(q, v)


class BasePolicy:
    def __init__(self, model_path, policy_action_scale=None):
        self.robot_config = Robot
        self.observation_config = Observation
        self.policy_config = Policy
        
        self.policy_action_scale = policy_action_scale or self.robot_config.policy_action_scale
        
        self.num_dofs = self.robot_config.num_joints
        self.default_dof_angles = np.array(self.robot_config.default_dof_angles)
        
        self.obs_scales = self.observation_config.obs_scales
        self.obs_dims = self.observation_config.obs_dims
        self.obs_dict = self.policy_config.obs_dict_base
        self.history_length_dict = self.observation_config.history_length_dict
        
        self.obs_dim_dict = self._calculate_obs_dim_dict()
        self.motor_effort_limits = np.array(self.robot_config.motor_effort_limits)
        
        self.obs_buf_dict = {
            key: np.zeros((1, self.obs_dim_dict[key] * self.history_length_dict[key])) 
            for key in self.obs_dim_dict
        }
        
        self.last_policy_action = np.zeros((1, self.num_dofs))
        
        self.setup_policy(model_path)
    
    def _calculate_obs_dim_dict(self):
        return self.policy_config.get_obs_dim_dict(self.obs_dict, self.obs_dims)
    
    def setup_policy(self, model_path):
        self.onnx_policy_session = onnxruntime.InferenceSession(model_path)
        input_names = [inp.name for inp in self.onnx_policy_session.get_inputs()]
        output_names = [out.name for out in self.onnx_policy_session.get_outputs()]
        
        self.onnx_input_names = input_names
        self.onnx_output_names = output_names
        
        def policy_act(obs_dict):
            input_feed = {name: obs_dict[name] for name in self.onnx_input_names}
            outputs = self.onnx_policy_session.run(self.onnx_output_names, input_feed)
            return outputs[0]

        self.policy = policy_act
    
    def _convert_vuer_state_to_robot_data(self, qpos, qvel):
        robot_state_data = np.concatenate([qpos, qvel]).reshape(1, -1)
        return robot_state_data
    
    def get_current_obs_buffer_dict(self, robot_state_data):
        current_obs_buffer_dict = {}

        # Extract base and joint data
        current_obs_buffer_dict["base_quat"] = robot_state_data[:, 3:7]  # MuJoCo format: [w,x,y,z]
        current_obs_buffer_dict["base_ang_vel"] = robot_state_data[:, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6]
        current_obs_buffer_dict["dof_pos"] = robot_state_data[:, 7 : 7 + self.num_dofs] - self.default_dof_angles
        current_obs_buffer_dict["dof_vel"] = robot_state_data[:, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs]

        # Calculate projected gravity
        v = np.array([[0, 0, -1]])
        current_obs_buffer_dict["projected_gravity"] = quat_rotate_inverse_numpy(
            current_obs_buffer_dict["base_quat"], v
        )
        
        current_obs_buffer_dict["base_lin_vel"] = np.zeros((1, 3))
        current_obs_buffer_dict["command_lin_vel"] = np.zeros((1, 2))
        current_obs_buffer_dict["command_ang_vel"] = np.zeros((1, 1))
        current_obs_buffer_dict["command_stand"] = np.ones((1, 1))
        current_obs_buffer_dict["command_base_height"] = np.ones((1, 1)) * self.robot_config.desired_base_height
        current_obs_buffer_dict["command_waist_dofs"] = np.zeros((1, 3))
        current_obs_buffer_dict["ref_upper_dof_pos"] = np.zeros((1, 14))
        current_obs_buffer_dict["actions"] = self.last_policy_action
        current_obs_buffer_dict["phase_time"] = np.zeros((1, 1))

        return current_obs_buffer_dict
    
    def group_and_scale_observations(self, individual_obs):
        """Group individual observations and apply scaling for policy input.
        
        Args:
            individual_obs: Dict of individual observation components (e.g., "dof_pos", "dof_vel")
            
        Returns:
            Dict of grouped observations ready for policy (e.g., "actor_obs")
        """
        grouped_obs = {}
        for group_name in self.obs_dict:
            component_names = sorted(self.obs_dict[group_name])
            scaled_components = [individual_obs[name] * self.obs_scales[name] for name in component_names]
            grouped_obs[group_name] = np.concatenate(scaled_components, axis=1)
        return grouped_obs
    
    def prepare_obs_for_rl(self, robot_state_data):
        individual_obs = self.get_current_obs_buffer_dict(robot_state_data)
        grouped_obs = self.group_and_scale_observations(individual_obs)

        # print(individual_obs)

        self.obs_buf_dict = {
            key: np.concatenate(
                (
                    self.obs_buf_dict[key][:, self.obs_dim_dict[key] : (self.obs_dim_dict[key] * self.history_length_dict[key])],
                    grouped_obs[key],
                ),
                axis=1,
            )
            for key in self.obs_buf_dict
        }

        return {"actor_obs": self.obs_buf_dict["actor_obs"].astype(np.float32)}
    
    def predict(self, qpos, qvel):
        robot_state_data = self._convert_vuer_state_to_robot_data(qpos, qvel)
        
        obs = self.prepare_obs_for_rl(robot_state_data)
        
        policy_action = self.policy(obs)
        policy_action = np.clip(policy_action, -100, 100)
        
        self.last_policy_action = policy_action.copy()
        
        scaled_action = policy_action * self.policy_action_scale
        q_target = scaled_action.flatten() + np.array(self.robot_config.default_dof_angles)
        
        torques = self.pd_control(q_target, qpos, qvel)
        
        return torques
    
    def pd_control(self, q_target, qpos, qvel):
        current_joint_pos = qpos[7:]  # skip base pose (7 DOF)
        current_joint_vel = qvel[6:]  # skip base velocity (6 DOF)

        # q_target = [
        #     -0.1,  # left_hip_yaw_joint
        #     0.0,   # left_hip_roll_joint
        #     0.0,   # left_hip_pitch_joint
        #     0.3,   # left_knee_joint
        #     -0.2,  # left_ankle_pitch_joint
        #     0.0,   # left_ankle_roll_joint
        #     -0.1,  # right_hip_yaw_joint
        #     0.0,   # right_hip_roll_joint
        #     0.0,   # right_hip_pitch_joint
        #     0.3,   # right_knee_joint
        #     -0.2,  # right_ankle_pitch_joint
        #     0.0,   # right_ankle_roll_joint
        #     0.0,   # waist_yaw_joint
        #     0.0,   # waist_roll_joint
        #     0.0,   # waist_pitch_joint
        #     0.0,   # left_shoulder_pitch_joint
        #     0.0,   # left_shoulder_roll_joint
        #     0.0,   # left_shoulder_yaw_joint
        #     0.0,   # left_elbow_joint
        #     0.0,   # left_wrist_roll_joint
        #     0.0,   # left_wrist_pitch_joint
        #     0.0,   # left_wrist_yaw_joint
        #     0.0,   # right_shoulder_pitch_joint
        #     0.0,   # right_shoulder_roll_joint
        #     0.0,   # right_shoulder_yaw_joint
        #     0.0,   # right_elbow_joint
        #     0.0,   # right_wrist_roll_joint
        #     0.0,   # right_wrist_pitch_joint
        #     0.0    # right_wrist_yaw_joint
        # ]

        position_error = q_target - current_joint_pos
        velocity_error = 0.0 - current_joint_vel

        # print("q target", q_target[:3])
        # print("current joint pos", current_joint_pos[:3])
        # print("error! ", position_error[:3])

        kp = np.array(self.robot_config.joint_kp)
        kd = np.array(self.robot_config.joint_kd)

        torques = kp * position_error + kd * velocity_error

        torques = np.clip(torques, -self.motor_effort_limits, self.motor_effort_limits)

        return torques
    
    def reset(self):
        self.obs_buf_dict = {
            key: np.zeros((1, self.obs_dim_dict[key] * self.history_length_dict[key])) 
            for key in self.obs_dim_dict
        }
        self.last_policy_action = np.zeros((1, self.num_dofs))