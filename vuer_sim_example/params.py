from params_proto.proto import Flag, PrefixProto

class RobotConfig(PrefixProto):
    robot_type = "g1_29dof"
    num_joints = 29
    num_motors = 29
    num_upper_body_joints = 14
    desired_base_height = 0.75
    policy_action_scale = 0.25
    gait_period = 0.9
    simulate_dt = 0.005
    viewer_dt = 0.02

    dof_names = [
        'left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint', 
        'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
        'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint',
        'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
        'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
        'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 
        'left_shoulder_yaw_joint', 'left_elbow_joint',
        'left_wrist_roll_joint', 'left_wrist_pitch_joint', 
        'left_wrist_yaw_joint',
        'right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
        'right_shoulder_yaw_joint', 'right_elbow_joint',
        'right_wrist_roll_joint', 'right_wrist_pitch_joint',
        'right_wrist_yaw_joint'
    ]
    
    upper_dof_names = [
        'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 
        'left_shoulder_yaw_joint', 'left_elbow_joint',
        'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
        'right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
        'right_shoulder_yaw_joint', 'right_elbow_joint',
        'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint'
    ]
    
    lower_dof_names = [
        'left_hip_yaw_joint', 'left_hip_roll_joint', 'left_hip_pitch_joint',
        'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
        'right_hip_yaw_joint', 'right_hip_roll_joint', 'right_hip_pitch_joint',
        'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
        'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint'
    ]
    
    default_dof_angles = [
        -0.1,  # left_hip_yaw_joint 
        0.0,   # left_hip_roll_joint
        0.0,   # left_hip_pitch_joint
        0.3,   # left_knee_joint
        -0.2,  # left_ankle_pitch_joint
        0.0,   # left_ankle_roll_joint
        -0.1,  # right_hip_yaw_joint
        0.0,   # right_hip_roll_joint
        0.0,   # right_hip_pitch_joint
        0.3,   # right_knee_joint
        -0.2,  # right_ankle_pitch_joint
        0.0,   # right_ankle_roll_joint
        0.0,   # waist_yaw_joint
        0.0,   # waist_roll_joint
        0.0,   # waist_pitch_joint
        0.0,   # left_shoulder_pitch_joint
        0.0,   # left_shoulder_roll_joint
        0.0,   # left_shoulder_yaw_joint
        0.0,   # left_elbow_joint
        0.0,   # left_wrist_roll_joint
        0.0,   # left_wrist_pitch_joint
        0.0,   # left_wrist_yaw_joint
        0.0,   # right_shoulder_pitch_joint
        0.0,   # right_shoulder_roll_joint
        0.0,   # right_shoulder_yaw_joint
        0.0,   # right_elbow_joint
        0.0,   # right_wrist_roll_joint
        0.0,   # right_wrist_pitch_joint
        0.0    # right_wrist_yaw_joint
    ]
    
    joint_kp = [
        100, 100, 100, 200, 20, 20,
        100, 100, 100, 200, 20, 20,
        300, 300, 300,
        90, 60, 20, 60, 4, 4, 4,
        90, 60, 20, 60, 4, 4, 4
    ]
    
    joint_kd = [
        2.5, 2.5, 2.5, 5, 0.2, 0.1,
        2.5, 2.5, 2.5, 5, 0.2, 0.1,
        5.0, 5.0, 5.0,
        2.0, 1.0, 0.4, 1.0, 0.2, 0.2, 0.2,
        2.0, 1.0, 0.4, 1.0, 0.2, 0.2, 0.2
    ]
    
    motor_pos_lower_limits = [
        -2.5307, -0.5236, -2.7576, -0.087267, -0.87267, -0.2618, 
        -2.5307, -2.9671, -2.7576, -0.087267, -0.87267, -0.2618, 
        -2.618, -0.52, -0.52,
        -3.0892, -1.5882, -2.618, -1.0472, 
        -1.972222054, -1.61443, -1.61443,
        -3.0892, -2.2515, -2.618, -1.0472, 
        -1.972222054, -1.61443, -1.61443
    ]
    
    motor_pos_upper_limits = [
        2.8798, 2.9671, 2.7576, 2.8798, 0.5236, 0.2618, 
        2.8798, 0.5236, 2.7576, 2.8798, 0.5236, 0.2618, 
        2.618, 0.52, 0.52,
        2.6704, 2.2515, 2.618, 2.0944, 
        1.972222054, 1.61443, 1.61443,
        2.6704, 1.5882, 2.618, 2.0944, 
        1.972222054, 1.61443, 1.61443
    ]

    motor_effort_limits = [88.0, 88.0, 88.0, 139.0, 50.0, 50.0,
                              88.0, 88.0, 88.0, 139.0, 50.0, 50.0,
                              88.0, 50.0, 50.0,
                              25.0, 25.0, 25.0, 25.0,
                              25.0, 5.0, 5.0,
                              25.0, 25.0, 25.0, 25.0,
                              25.0, 5.0, 5.0]
    
    motor_to_joint = list(range(29))
    joint_to_motor = list(range(29))
    
    left_hand_link_name = "left_rubber_hand"
    right_hand_link_name = "right_rubber_hand"
    
    residual_upper_body_action = Flag("Enable residual upper body actions")
    use_upper_body_controller = Flag("Enable upper body IK controller")
    
    @property
    def upper_dof_indices(self):
        return [self.dof_names.index(dof) for dof in self.upper_dof_names]
    
    @property
    def lower_dof_indices(self):
        return [self.dof_names.index(dof) for dof in self.lower_dof_names]

class ObservationConfig(PrefixProto):
    base_lin_vel_dim = 3
    base_ang_vel_dim = 3
    projected_gravity_dim = 3
    command_lin_vel_dim = 2
    command_ang_vel_dim = 1
    command_stand_dim = 1
    command_base_height_dim = 1
    command_waist_dofs_dim = 3
    ref_upper_dof_pos_dim = 14  # upper body actions
    dof_pos_dim = 29
    dof_vel_dim = 29
    actions_dim = 29  # lower body actions
    phase_time_dim = 1
    
    base_lin_vel_scale = 2.0
    base_ang_vel_scale = 0.25
    projected_gravity_scale = 1.0
    command_lin_vel_scale = 1.0
    command_ang_vel_scale = 1.0
    command_stand_scale = 1.0
    command_base_height_scale = 2.0
    command_waist_dofs_scale = 1.0
    ref_upper_dof_pos_scale = 1.0
    dof_pos_scale = 1.0
    dof_vel_scale = 0.05
    history_scale = 1.0
    actions_scale = 1.0
    phase_time_scale = 1.0
    
    actor_obs_history = 5
    
    @property
    def obs_dims(self):
        return {
            "base_lin_vel": self.base_lin_vel_dim,
            "base_ang_vel": self.base_ang_vel_dim,
            "projected_gravity": self.projected_gravity_dim,
            "command_lin_vel": self.command_lin_vel_dim,
            "command_ang_vel": self.command_ang_vel_dim,
            "command_stand": self.command_stand_dim,
            "command_base_height": self.command_base_height_dim,
            "command_waist_dofs": self.command_waist_dofs_dim,
            "ref_upper_dof_pos": self.ref_upper_dof_pos_dim,
            "dof_pos": self.dof_pos_dim,
            "dof_vel": self.dof_vel_dim,
            "actions": self.actions_dim,
            "phase_time": self.phase_time_dim
        }
    
    @property
    def obs_scales(self):
        return {
            "base_lin_vel": self.base_lin_vel_scale,
            "base_ang_vel": self.base_ang_vel_scale,
            "projected_gravity": self.projected_gravity_scale,
            "command_lin_vel": self.command_lin_vel_scale,
            "command_ang_vel": self.command_ang_vel_scale,
            "command_stand": self.command_stand_scale,
            "command_base_height": self.command_base_height_scale,
            "command_waist_dofs": self.command_waist_dofs_scale,
            "ref_upper_dof_pos": self.ref_upper_dof_pos_scale,
            "dof_pos": self.dof_pos_scale,
            "dof_vel": self.dof_vel_scale,
            "history": self.history_scale,
            "actions": self.actions_scale,
            "phase_time": self.phase_time_scale
        }
    
    @property
    def history_length_dict(self):
        return {
            "actor_obs": self.actor_obs_history
        }

class PolicyConfig(PrefixProto):
    actor_obs_groups = [
        "base_ang_vel",
        "projected_gravity", 
        "command_lin_vel",
        "command_ang_vel",
        "command_stand",
        "command_base_height",
        "command_waist_dofs",
        "ref_upper_dof_pos",
        "dof_pos",
        "dof_vel", 
        "actions"
    ]
    
    base_obs_groups = [
        "base_ang_vel",
        "projected_gravity",
        "dof_pos",
        "dof_vel"
    ]
    
    @property
    def obs_dict_full(self):
        return {"actor_obs": self.actor_obs_groups}
    
    @property
    def obs_dict_base(self):
        return {"actor_obs": self.base_obs_groups}
    
    @property 
    def obs_dict_loco_manip(self):
        return self.obs_dict_full
    
    def get_obs_dim_dict(self, obs_dict, obs_dims):
        obs_dim_dict = {}
        for key in obs_dict:
            obs_dim_dict[key] = 0
            for obs_name in obs_dict[key]:
                if obs_name in obs_dims:
                    obs_dim_dict[key] += obs_dims[obs_name]
                else:
                    print(f"Warning: observation '{obs_name}' not found in obs_dims")
        return obs_dim_dict

Robot = RobotConfig()
Observation = ObservationConfig() 
Policy = PolicyConfig()