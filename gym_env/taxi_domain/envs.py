import os

import gym
import numpy as np
from gym.spaces.box import Box
from gym.wrappers import Monitor

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

try:
    import dm_control2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass


def make_env(env_id, seed, rank):
    def _thunk():
        env = gym.make(env_id,  task_num=rank)
        env.seed(seed + rank)
        return env

    return _thunk


class AddTimestep(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AddTimestep, self).__init__(env)
        self.observation_space = Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            [self.observation_space.shape[0] + 1],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return np.concatenate((observation, [self.env._elapsed_steps]))


class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        if len(obs_shape) == 1:
            self.observation_space = Box(
                self.observation_space.low[0],
                self.observation_space.high[0],
                [obs_shape[0]],
                dtype=self.observation_space.dtype)
        elif len(obs_shape) == 3:
            self.observation_space = Box(
                self.observation_space.low[0, 0, 0],
                self.observation_space.high[0, 0, 0],
                [obs_shape[2], obs_shape[0], obs_shape[1]],
                dtype=self.observation_space.dtype)
        else:
            raise NotImplementedError

    def observation(self, observation):
        return observation.transpose(2, 0, 1)