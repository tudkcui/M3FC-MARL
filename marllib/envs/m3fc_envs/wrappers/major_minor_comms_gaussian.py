import numpy as np
from gym.spaces import Box, Tuple

from marllib.envs.m3fc_envs.major_minor_mf_marl_env import MajorMinorMARLEnv
from marllib.envs.m3fc_envs.wrappers.major_minor_discrete_gaussian import DiscreteGaussianMajorMinorMFCEnv


class CommsGaussianMajorMinorMFCEnv(DiscreteGaussianMajorMinorMFCEnv):

    """
    Wraps a spatial MFMARL problem with continuous state and continuous action space.
    """
    def __init__(self, env_MARL: MajorMinorMARLEnv, x_bins_per_dim: int = 5, **kwargs):
        super().__init__(env_MARL, x_bins_per_dim, **kwargs)

        self.num_obs_layers = 1 + self.num_dims_time + 1
        self.observation_space = Tuple([Box(0, 1, shape=(x_bins_per_dim,) * self.x_dims + (self.num_obs_layers,), dtype=np.float64),
                                        env_MARL.major_observation_space])

    def get_observation(self):
        xs, y = self.env_MARL.get_observation()
        obs_agents = self.get_histogram_from(xs)
        obs_users = self.get_histogram_from(self.env_MARL.nu, normalizer=self.env_MARL.num_users)
        if self.obs_time:
            obs_time = np.tile(self.get_time_obs(), (self.x_bins_per_dim,) * self.x_dims + (1,))
            return np.concatenate([obs_agents, obs_users, obs_time], axis=-1), y
        else:
            return np.concatenate([obs_agents, obs_users], axis=-1), y
