import numpy as np
from gym.spaces import Box

from marllib.envs.m3fc_envs.wrappers.discrete_gaussian import DiscreteGaussianMFCEnv


class CommsObsGaussianWrapperPerAgent(DiscreteGaussianMFCEnv):

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

    def sample_actions(self, action):
        x_idxs = self.get_bin_index_from_position(self.env_MARL.xs)
        actions_minor = np.zeros((self.env_MARL.num_agents, self.u_dims))
        for i in range(self.env_MARL.num_agents):
            action_minor = action[i]
            if self.unbounded_actions:
                action_minor = np.tanh(action_minor)

            action_minor = np.reshape(action_minor, self.true_action_shape)
            mean = (action_minor[tuple(x_idxs)][i, ..., 0] + 1) / 2 * (self.u_high - self.u_low) + self.u_low
            cov = (action_minor[tuple(x_idxs)][i, ..., 1] + 1 + 1e-10) / 4
            actions_minor[i] = np.clip(np.random.normal(mean, cov), self.u_low, self.u_high)
        return actions_minor
