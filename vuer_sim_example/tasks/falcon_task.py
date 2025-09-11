import numpy as np
from typing import Optional
from .base_task import Task

class FalconTask(Task):
    def __init__(self):
        self.default_angles = np.zeros(29)
        self.commands = np.array([1.0, 0.0, 0.0])
        self.phase = 0.0
        self.last_actions = np.zeros(29)
        
    def quat_rotate_inverse(self, quat, vec):
        q = np.array(quat)
        q_conj = np.array([-q[0], -q[1], -q[2], q[3]])
        vec_quat = np.array([vec[0], vec[1], vec[2], 0.0])
        
        def quat_mult(q1, q2):
            w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
            w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]
            return np.array([
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2,
                w1*w2 - x1*x2 - y1*y2 - z1*z2
            ])
        
        result = quat_mult(quat_mult(q_conj, vec_quat), q)
        return result[:3]
    
    def compute_obs(self, physics) -> np.ndarray:
        state = physics.get_state()
        qpos = np.array(state.get("qpos", []))
        qvel = np.array(state.get("qvel", []))
        
        if len(qpos) == 0 or len(qvel) == 0:
            return np.array([])
        
        # falcon observation processing
        dof_pos = qpos[7:] - self.default_angles[:len(qpos[7:])]
        dof_vel = qvel[6:] * 0.05
        base_quat = qpos[3:7]
        projected_gravity = self.quat_rotate_inverse(base_quat, [0, 0, -1])
        base_ang_vel = qvel[3:6] * 0.25
        
        # update phase
        self.phase += 0.02
        sin_cos_phase = np.array([np.sin(self.phase), np.cos(self.phase)])
        
        # build observation
        obs = np.concatenate([
            dof_pos, dof_vel, projected_gravity, base_ang_vel,
            self.commands, self.last_actions[:len(dof_pos)], sin_cos_phase
        ])
        
        return obs
    
    def update_commands(self, physics) -> Optional[np.ndarray]:
        return None