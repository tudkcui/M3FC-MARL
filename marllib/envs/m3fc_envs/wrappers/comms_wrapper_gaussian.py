import numpy as np
from gym.spaces import Box

from marllib.envs.m3fc_envs.wrappers.discrete_gaussian import DiscreteGaussianMFCEnv


class CommsObsGaussianWrapper(DiscreteGaussianMFCEnv):

    def __init__(self, env_MARL, x_bins_per_dim: int = 5, stretch_factor: float = 10, **kwargs):
        super().__init__(env_MARL, x_bins_per_dim, periodic_time=False, **kwargs)

        self.stretch_factor = stretch_factor
        self.num_obs_layers = 1 + self.num_dims_time + 1
        self.observation_space = Box(0, self.stretch_factor, shape=(x_bins_per_dim,) * self.x_dims + (self.num_obs_layers,), dtype=np.float64)

    def get_mfc_observation(self):
        obs_agents = self.get_histogram_from(self.env_MARL.xs)
        obs_users = self.get_histogram_from(self.env_MARL.nu, normalizer=self.env_MARL.num_users)
        if self.obs_time:
            obs_time = np.tile(self.get_time_obs(), (self.x_bins_per_dim,) * self.x_dims + (1,))
            return self.stretch_factor * np.concatenate([obs_agents, obs_users, obs_time], axis=-1)
        else:
            return self.stretch_factor * np.concatenate([obs_agents, obs_users], axis=-1)
