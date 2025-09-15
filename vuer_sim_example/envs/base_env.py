import numpy as np
import asyncio
from typing import Optional
from vuer_sim_example.tasks.base_task import Task

class BaseEnv:
    def __init__(self, physics, task: Task):
        self.physics = physics
        self.task = task

    def start(self):
        self.physics.start()

    def reset(self):
        async def _async_reset():
            await self.physics.reset()
            self.task.reset(self.physics)
            return self.task.compute_obs(self.physics)
        
        return asyncio.run(_async_reset())

    def step(self, action: Optional[np.ndarray] = None):
        async def _async_step():
            nonlocal action
            if action is None:
                action = self.task.update_commands(self.physics)

            await self.physics.step(action)

            obs = self.task.compute_obs(self.physics)
            reward = self.task.compute_rew(self.physics)
            done = self.task.is_done(self.physics)
            info = self.task.get_info(self.physics)
            
            return obs, reward, done, info
        
        return asyncio.run(_async_step())

    def unwrapped(self):
        return self

    @property
    def action_space(self):
        class MockActionSpace:
            def sample(self):
                return np.random.randn(35)
        return MockActionSpace()