import numpy as np
from gym.spaces import Box, Tuple

from marllib.envs.m3fc_envs.major_minor_mf_marl_env import MajorMinorMARLEnv
from marllib.envs.m3fc_envs.wrappers.major_minor_discrete_gaussian import DiscreteGaussianMajorMinorMFCEnv


class CommsGaussianMajorMinorMFCEnvPerAgent(DiscreteGaussianMajorMinorMFCEnv):

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

    def sample_actions(self, action):
        action_major = action[0][1]

        x_idxs = self.get_bin_index_from_position(self.env_MARL.xs)
        actions_minor = np.zeros((self.env_MARL.num_agents, self.u_dims))
        for i in range(self.env_MARL.num_agents):
            action_minor = action[i][0]

            if self.unbounded_actions:
                action_minor = np.tanh(action_minor)

            action_minor = np.reshape(action_minor, self.true_action_shape)
            mean = (action_minor[tuple(x_idxs)][i, ..., 0] + 1) / 2 * (self.u_high - self.u_low) + self.u_low
            cov = (action_minor[tuple(x_idxs)][i, ..., 1] + 1 + 1e-10) / 4
            actions_minor[i] = np.clip(np.random.normal(mean, cov), self.u_low, self.u_high)

        return actions_minor, action_major
