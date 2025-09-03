import numpy as np
from gym_dmc.gym.core import Wrapper


class HistoryWrapper(Wrapper):
    def __init__(self, env, *, history_len=10):
        super().__init__(env)
        self.env = env

        self.step_counter = 0

        self.num_obs = 7
        self.num_history = history_len
        self.history_buf = np.zeros((history_len, self.num_obs))

    def step(self, action):
        # privileged information and observation history are stored in info
        obs, rew, done, info = self.env.step(action)
        observations = obs["observations"]
        obs_np = np.array(observations)

        self.step_counter += 1

        if self.step_counter < 2:
            # repeat obs to fill history buffer
            obs_history = np.repeat(obs_np, self.num_history + 1, axis=0)
            self.history_buf = np.repeat(obs_np, self.num_history, axis=0)
        else:
            obs_history = np.concatenate([obs_np, self.history_buf], axis=0)
            self.history_buf = np.concatenate([self.history_buf[1:, :], obs_np], axis=0)

        # add the privileged obs

        obs["observations"] = obs_history
        return obs, rew, done, info

    def reset(self):
        obs = super().reset()
        observations = obs["observations"]

        # repeat obs to fill history buffer
        obs_np = np.array(observations)

        obs_history = np.repeat(obs_np, self.num_history + 1, axis=0)
        self.history_buf = np.repeat(obs_np, self.num_history, axis=0)

        # concatenate one more
        obs["observations"] = obs_history
        return obs
