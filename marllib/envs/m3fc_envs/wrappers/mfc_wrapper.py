from abc import ABC, abstractmethod

import gym
import numpy as np
from gym.spaces import Box, Tuple

from marllib.envs.m3fc_envs.mf_marl_env import MeanFieldMARLEnv


class MFCEnv(gym.Env, ABC):

    """
    Wraps a MFMARL problem as a single-agent MFC MDP.
    """
    def __init__(self, env_MARL: MeanFieldMARLEnv, obs_time=0, periodic_time=False, **kwargs, ):
        self.env_MARL = env_MARL
        self.x_dims = self.env_MARL.observation_space.shape[0] if isinstance(self.env_MARL.observation_space, Box) \
            else len(self.env_MARL.observation_space) if isinstance(self.env_MARL.observation_space, Tuple) else 1
        self.u_dims = self.env_MARL.action_space.shape[0] if isinstance(self.env_MARL.action_space, Box) \
            else len(self.env_MARL.action_space) if isinstance(self.env_MARL.action_space, Tuple) else 1
        self.x_low = self.env_MARL.observation_space.low if hasattr(self.env_MARL.observation_space, 'low') else None
        self.x_high = self.env_MARL.observation_space.high if hasattr(self.env_MARL.observation_space, 'high') else None
        self.u_low = self.env_MARL.action_space.low if hasattr(self.env_MARL.action_space, 'low') else None
        self.u_high = self.env_MARL.action_space.high if hasattr(self.env_MARL.action_space, 'high') else None
        self.obs_time = obs_time >= 1
        self.one_hot_time = obs_time >= 2
        self.periodic_time = periodic_time

        self.num_dims_time = 0 if not self.obs_time \
            else 1 if not self.one_hot_time \
            else self.env_MARL.period_time if self.periodic_time \
            else self.env_MARL.time_steps

        self.num_agents = 1



    def step(self, action):
        actions = self.sample_actions(action)
        xs, reward, dones, _ = self.env_MARL.step(actions)
        return self.get_mfc_observation(), reward, dones['__all__'], {}

    @abstractmethod
    def sample_actions(self, action):
        pass

    @abstractmethod
    def get_mfc_observation(self):
        pass

    def get_time_obs(self):
        obs = np.zeros((self.num_dims_time,))
        if self.obs_time:
            if self.one_hot_time:
                obs[self.env_MARL.t % self.num_dims_time] = 1
            else:
                obs[0] = (self.env_MARL.t % self.num_dims_time) / self.num_dims_time
        return obs

    def reset(self, **kwargs):
        self.env_MARL.reset()
        return self.get_mfc_observation()

    def render(self, mode="human"):
        pass
