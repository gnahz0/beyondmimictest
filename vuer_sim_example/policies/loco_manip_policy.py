import numpy as np
from .base_policy import BasePolicy

class LocoManipPolicy(BasePolicy):
    def __init__(self, model_path, policy_action_scale=None):
        super().__init__(model_path, policy_action_scale)

        print("Inputs:", getattr(self.policy, "onnx_input_names", None))
        print("Outputs:", getattr(self.policy, "onnx_output_names", None))

        self.obs_dict = self.policy_config.obs_dict_loco_manip
        self.obs_dim_dict = self._calculate_obs_dim_dict()
        
        # reinitialize observation buffers
        self.obs_buf_dict = {
            key: np.zeros((1, self.obs_dim_dict[key] * self.history_length_dict[key])) 
            for key in self.obs_dim_dict
        }
        
        self.num_upper_dofs = self.robot_config.num_upper_body_joints
        self.residual_upper_body_action = self.robot_config.residual_upper_body_action
        
        self.upper_dof_indices = self.robot_config.upper_dof_indices
        self.lower_dof_indices = self.robot_config.lower_dof_indices
        
        self.ref_upper_dof_pos = np.zeros((1, self.num_upper_dofs))
        if self.upper_dof_indices:
            self.ref_upper_dof_pos += np.array(self.default_dof_angles)[self.upper_dof_indices]
        
        self.base_height_command = np.array([[self.robot_config.desired_base_height]])
        self.command_lin_vel = np.zeros((1, 2))
        self.command_ang_vel = np.zeros((1, 1))
        self.command_stand = np.zeros((1, 1))
        self.command_waist_dofs = np.zeros((1, 3))
    
    def get_current_obs_buffer_dict(self, robot_state_data):
        current_obs_dict = super().get_current_obs_buffer_dict(robot_state_data)
        
        current_obs_dict["command_lin_vel"] = self.command_lin_vel
        current_obs_dict["command_ang_vel"] = self.command_ang_vel
        current_obs_dict["command_stand"] = self.command_stand
        current_obs_dict["command_base_height"] = self.base_height_command
        current_obs_dict["command_waist_dofs"] = self.command_waist_dofs
        current_obs_dict["ref_upper_dof_pos"] = self.ref_upper_dof_pos
        current_obs_dict["actions"] = self.last_policy_action
        
        return current_obs_dict
    
    def predict(self, qpos, qvel, gamepad=None):
        robot_state_data = self._convert_vuer_state_to_robot_data(qpos, qvel)

        if gamepad:
            print(gamepad)
            if gamepad['buttons'][5]:
                self.command_stand = 1 - self.command_stand
            self.command_lin_vel[0][0] = -gamepad['axes'][1]
            self.command_lin_vel[0][1] = -gamepad['axes'][0]
            self.command_ang_vel[0][0] = -gamepad['axes'][2]
        
        obs = self.prepare_obs_for_rl(robot_state_data)

        # >>> ADD THIS: adapt to the model's input names
        onnx_inputs = getattr(self, "onnx_input_names", [])
        if "obs" in onnx_inputs and "actor_obs" in obs:
            obs["obs"] = obs["actor_obs"].astype(np.float32)

        if "time_step" in onnx_inputs:
            if not hasattr(self, "_step"):
                self._step = 0
            # use a simple counter; replace with phase if you trained that way
            obs["time_step"] = np.array([[self._step]], dtype=np.float32)
            self._step += 1
        # <<< END ADD

        policy_action = self.policy(obs)
        policy_action = np.clip(policy_action, -100, 100)
        
        # update tracking
        self.last_policy_action = policy_action.copy()
        
        # scale policy action
        scaled_action = policy_action * self.policy_action_scale
        q_target = scaled_action.flatten() + np.array(self.robot_config.default_dof_angles)
        
        # residual upper body action
        if self.residual_upper_body_action and self.upper_dof_indices:
            upper_residual = (self.ref_upper_dof_pos - np.array(self.default_dof_angles)[self.upper_dof_indices]).flatten()
            q_target[self.upper_dof_indices] += upper_residual

        torques = self.pd_control(q_target, qpos, qvel)

        # NOTE: for free base
        torques = np.concatenate((np.zeros(6), torques))
        
        return torques
    
    def set_base_height_command(self, height):
        self.base_height_command[0, 0] = height
    
    def set_upper_body_reference(self, ref_positions):
        if len(ref_positions) == self.num_upper_dofs:
            self.ref_upper_dof_pos[0, :] = ref_positions
        else:
            raise ValueError(f"Expected {self.num_upper_dofs} upper body positions, got {len(ref_positions)}")
    
    def reset(self):
        super().reset()
        
        if self.upper_dof_indices:
            self.ref_upper_dof_pos = np.zeros((1, self.num_upper_dofs))
            self.ref_upper_dof_pos += self.default_dof_angles[self.upper_dof_indices]
        
        self.base_height_command = np.array([[self.robot_config.desired_base_height]])