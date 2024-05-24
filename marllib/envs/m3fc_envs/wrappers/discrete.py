from abc import ABC

import numpy as np
from gym.spaces import Box

from marllib.envs.m3fc_envs.mf_marl_env import MeanFieldMARLEnv
from marllib.envs.m3fc_envs.wrappers.mfc_wrapper import MFCEnv


class StateDiscretizedMFCEnv(MFCEnv, ABC):

    """
    Wraps a spatial MFMARL problem with continuous state and continuous action space.
    Describes an MFMDP through discretized states and to-be-implemented actions.
    """
    def __init__(self, env_MARL: MeanFieldMARLEnv, x_bins_per_dim: int = 5, **kwargs):
        super().__init__(env_MARL, **kwargs)

        self.x_bins_per_dim = x_bins_per_dim
        self.num_obs_layers = 1 + self.num_dims_time
        self.observation_space = Box(0, 1, shape=(x_bins_per_dim,) * self.x_dims + (self.num_obs_layers,), dtype=np.float64)
        self.x_bin_centers = np.array([self.x_low + (self.x_high - self.x_low) / (2 * x_bins_per_dim)
                                       + (self.x_high - self.x_low) / x_bins_per_dim * i for i in range(x_bins_per_dim)])

    def get_bin_index_from_position(self, xs):
        if len(xs) == 0:
            return []
        return np.array([np.abs(np.expand_dims(self.x_bin_centers[:, d], axis=1)
                                - np.expand_dims(xs[:, d], axis=0)).argmin(axis=0)
                         for d in range(self.x_dims)])

    def get_mfc_observation(self):
        obs_agents = self.get_histogram_from(self.env_MARL.xs)
        if self.obs_time:
            obs_time = np.tile(self.get_time_obs(), (self.x_bins_per_dim,) * self.x_dims + (1,))
            return np.concatenate([obs_agents, obs_time], axis=-1)
        else:
            return obs_agents

    def get_histogram_from(self, xs, normalizer=None):
        obs = np.zeros((self.x_bins_per_dim,) * self.x_dims + (1,))
        for x_idx in zip(*self.get_bin_index_from_position(xs)):
            obs[x_idx + (0,)] += 1
        obs /= self.env_MARL.num_agents if normalizer is None else normalizer
        return obs
