import numpy as np
import onnxruntime
from ..params import Robot, Observation, Policy


def quat_rotate_inverse_numpy(quat, vec):
    """Rotate vector by inverse of quaternion.
    
    Args:
        quat: quaternion array of shape (1, 4) - [x, y, z, w]
        vec: vector array of shape (1, 3) - [x, y, z]
    
    Returns:
        rotated vector of shape (1, 3)
    """
    q = quat[0]
    v = vec[0]
    
    q_conj = np.array([-q[0], -q[1], -q[2], q[3]])
    
    vec_quat = np.array([v[0], v[1], v[2], 0.0])
    
    def quat_mult(q1, q2):
        """Multiply two quaternions [x, y, z, w]."""
        x1, y1, z1, w1 = q1[0], q1[1], q1[2], q1[3]
        x2, y2, z2, w2 = q2[0], q2[1], q2[2], q2[3]
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
            w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y  
            w1*z2 + x1*y2 - y1*x2 + z1*w2,  # z
            w1*w2 - x1*x2 - y1*y2 - z1*z2   # w
        ])
    
    result = quat_mult(quat_mult(q_conj, vec_quat), q)
    
    return result[:3].reshape(1, 3)


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
        
        base_quat = robot_state_data[:, 3:7]
        current_obs_buffer_dict["base_ang_vel"] = robot_state_data[:, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6]
        current_obs_buffer_dict["dof_pos"] = robot_state_data[:, 7 : 7 + self.num_dofs] - self.default_dof_angles
        current_obs_buffer_dict["dof_vel"] = robot_state_data[:, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs]
        
        v = np.array([[0, 0, -1]])
        current_obs_buffer_dict["projected_gravity"] = quat_rotate_inverse_numpy(base_quat, v)
        
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
    
    def parse_current_obs_dict(self, current_obs_buffer_dict):
        # print(current_obs_buffer_dict)
        current_obs_dict = {}
        for key in self.obs_dict:
            obs_list = sorted(self.obs_dict[key])
            current_obs_dict[key] = np.concatenate(
                [current_obs_buffer_dict[obs_name] * self.obs_scales[obs_name] for obs_name in obs_list], axis=1
            )
        return current_obs_dict
    
    def prepare_obs_for_rl(self, robot_state_data):
        current_obs_buffer_dict = self.get_current_obs_buffer_dict(robot_state_data)
        current_obs_dict = self.parse_current_obs_dict(current_obs_buffer_dict)
        
        # update observation buffers with history
        self.obs_buf_dict = {
            key: np.concatenate(
                (
                    self.obs_buf_dict[key][:, self.obs_dim_dict[key] : (self.obs_dim_dict[key] * self.history_length_dict[key])],
                    current_obs_dict[key],
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

        # q_target = np.array([
        #     -0.1,  # left_hip_pitch_joint
        #     0.0,  # left_hip_roll_joint
        #     0.0,  # left_hip_yaw_joint
        #     0.3,  # left_knee_joint
        #     -0.2, # left_ankle_pitch_joint
        #     0.0,  # left_ankle_roll_joint
        #     -0.1, # right_hip_pitch_joint
        #     0.0,  # right_hip_roll_joint
        #     0.0,  # right_hip_yaw_joint
        #     0.3,  # right_knee_joint
        #     -0.2, # right_ankle_pitch_joint
        #     0.0,  # right_ankle_roll_joint
        #     0.0,  # waist_yaw_joint
        #     0.0,  # waist_roll_joint
        #     0.0,  # waist_pitch_joint
        #     0.0,  # left_shoulder_pitch_joint
        #     0.0,  # left_shoulder_roll_joint
        #     0.0,  # left_shoulder_yaw_joint
        #     0.0,  # left_elbow_joint
        #     0.0,  # left_wrist_roll_joint
        #     0.0,  # left_wrist_pitch_joint
        #     0.0,  # left_wrist_yaw_joint
        #     0.0,  # right_shoulder_pitch_joint
        #     0.0,  # right_shoulder_roll_joint
        #     0.0,  # right_shoulder_yaw_joint
        #     0.0,  # right_elbow_joint
        #     0.0,  # right_wrist_roll_joint
        #     0.0,  # right_wrist_pitch_joint
        #     0.0   # right_wrist_yaw_joint
        # ])

        position_error = q_target - current_joint_pos
        velocity_error = 0.0 - current_joint_vel
        
        kp = np.array(self.robot_config.joint_kp)
        kd = np.array(self.robot_config.joint_kd)
        
        torques = kp * position_error + kd * velocity_error

        torques = np.clip(torques, -60.0, 60.0)
        
        return torques
    
    def reset(self):
        self.obs_buf_dict = {
            key: np.zeros((1, self.obs_dim_dict[key] * self.history_length_dict[key])) 
            for key in self.obs_dim_dict
        }
        self.last_policy_action = np.zeros((1, self.num_dofs))