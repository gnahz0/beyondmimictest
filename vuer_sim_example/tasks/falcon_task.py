import numpy as np
from typing import Optional
from .base_task import Task
from ..policies.base_policy import quat_rotate_inverse_numpy

class FalconTask(Task):
    def __init__(self):
        self.default_angles = np.zeros(29)
        self.commands = np.array([1.0, 0.0, 0.0])
        self.phase = 0.0
        self.last_actions = np.zeros(29)

    def compute_obs(self, physics) -> np.ndarray:
        state = physics.get_state()
        qpos = np.array(state.get("qpos", []))
        qvel = np.array(state.get("qvel", []))
        
        if len(qpos) == 0 or len(qvel) == 0:
            return np.array([])
        
        dof_pos = qpos[7:] - self.default_angles[:len(qpos[7:])]
        dof_vel = qvel[6:] * 0.05
        base_quat = qpos[3:7]
        projected_gravity = quat_rotate_inverse_numpy(base_quat.reshape(1, -1), np.array([[0, 0, -1]])).flatten()
        base_ang_vel = qvel[3:6] * 0.25
        
        self.phase += 0.02
        sin_cos_phase = np.array([np.sin(self.phase), np.cos(self.phase)])
        
        obs = np.concatenate([
            dof_pos, dof_vel, projected_gravity, base_ang_vel,
            self.commands, self.last_actions[:len(dof_pos)], sin_cos_phase
        ])
        
        return obs
    
    def update_commands(self, physics) -> Optional[np.ndarray]:
        return None