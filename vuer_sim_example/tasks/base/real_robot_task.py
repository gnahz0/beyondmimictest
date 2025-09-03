from dm_env import specs
import numpy as np
from vuer_sim_example.tasks.base.real_robot_env import RealRobotEnv

class RealRobotTask:
    def __init__(self, robot_env: RealRobotEnv):
        self.env = robot_env
        self.step_counter = 0

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def observation_spec(self):
        return {
            "state": specs.Array(shape=(16,), dtype=np.float32, name="state"),
        }

    def action_spec(self):
        return specs.BoundedArray(
            shape=(16,),
            dtype=np.float32,
            minimum=-np.ones(16) * 10, # TODO: Adjust based on robot limits
            maximum=np.ones(16) * 10, # TODO: Adjust based on robot limits
            name="action",
        )

    def get_observation(self):
        return self.env.get_obs()

    def get_prev_action(self):
        return self.env.get_prev_action()

    def get_reward(self):
        return self.env.get_reward(self.env.get_obs())

    def get_termination(self):
        return self.env.is_done(self.env.get_obs())

    def get_info(self):
        return {}
