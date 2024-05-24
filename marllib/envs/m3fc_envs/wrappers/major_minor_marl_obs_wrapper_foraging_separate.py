import numpy as np
from gym.spaces import Box, Tuple
from ray.rllib import MultiAgentEnv

from marllib.envs.m3fc_envs.major_minor_mf_marl_env import MajorMinorMARLEnv


class MajorMinorMARLObsWrapperForagingSeparate(MultiAgentEnv):
    """
    Wraps a mean-field MARL problem in discrete time with extra observations.
    """
    def __init__(self, env_MARL: MajorMinorMARLEnv, x_bins_per_dim: int = 5, unbounded_actions=False,
                 obs_time=0, periodic_time=False, **kwargs, ):
        super().__init__()
        self.env_MARL = env_MARL
        self.x_dims = self.env_MARL.minor_observation_space.shape[0]
        self.u_dims = self.env_MARL.minor_action_space.shape[0]
        self.x_low = self.env_MARL.minor_observation_space.low
        self.x_high = self.env_MARL.minor_observation_space.high
        self.u_low = self.env_MARL.minor_action_space.low
        self.u_high = self.env_MARL.minor_action_space.high
        self.major_u_low = self.env_MARL.major_action_space.low
        self.major_u_high = self.env_MARL.major_action_space.high
        self.obs_time = obs_time >= 1
        self.one_hot_time = obs_time >= 2
        self.periodic_time = periodic_time
        self.unbounded_actions = unbounded_actions

        self.num_dims_time = 0 if not self.obs_time \
            else 1 if not self.one_hot_time \
            else self.env_MARL.period_time if self.periodic_time \
            else self.env_MARL.time_steps

        self._agent_ids = list(range(-1, self.env_MARL.num_agents))

        self.x_bins_per_dim = x_bins_per_dim
        self.num_obs_layers = 1 + self.num_dims_time + 1
        self.x_bin_centers = np.array([self.x_low + (self.x_high - self.x_low) / (2 * x_bins_per_dim)
                                       + (self.x_high - self.x_low) / x_bins_per_dim * i for i in range(x_bins_per_dim)]).transpose()
        self.x_bin_centers = list(self.x_bin_centers)
        self.x_bin_centers[self.x_dims-1] = np.array([0, 1])
        self.observation_space = Tuple([
            self.env_MARL.minor_observation_space,
            self.env_MARL.major_observation_space,
            Box(0, 1, shape=(np.prod((x_bins_per_dim,) * (self.x_dims - 1) + (2,) + (self.num_obs_layers,)).item(),), dtype=np.float64),
        ])
        self.action_space = self.env_MARL.minor_action_space  # Assuming same major and minor action space

    def get_bin_index_from_position(self, xs):
        if len(xs) == 0:
            return []
        return np.array([np.abs(np.expand_dims(self.x_bin_centers[d], axis=1)
                                - np.expand_dims(xs[:, d], axis=0)).argmin(axis=0)
                         for d in range(self.x_dims)])

    def get_histogram_from(self, xs, normalizer=None):
        obs = np.zeros((self.x_bins_per_dim,) * (self.x_dims - 1) + (2,) + (1,))
        for x_idx in zip(*self.get_bin_index_from_position(xs)):
            obs[x_idx + (0,)] += 1
        obs /= self.env_MARL.num_agents if normalizer is None else normalizer
        return obs

    def reset(self):
        self.env_MARL.reset()
        return self.get_observation()

    def get_observation(self):
        xs, y = self.env_MARL.get_observation()
        obs_agents = self.get_histogram_from(self.env_MARL.xs)
        obs_users = self.get_histogram_from(self.env_MARL.nu, normalizer=self.env_MARL.num_users)
        if self.obs_time:
            obs_time = np.tile(self.get_time_obs(), (self.x_bins_per_dim,) * (self.x_dims - 1) + (2,) + (1,))
            return {i: (xs[i], y, np.concatenate([obs_agents, obs_users, obs_time], axis=-1).flatten()) if i != -1 else
                    (0 * xs[0], y, np.concatenate([obs_agents, obs_users, obs_time], axis=-1).flatten())
                    for i in range(-1, self.env_MARL.num_agents)}
        else:
            return {i: (xs[i], y, np.concatenate([obs_agents, obs_users], axis=-1).flatten()) if i != -1 else
                    (0 * xs[0], y, np.concatenate([obs_agents, obs_users], axis=-1).flatten())
                    for i in range(-1, self.env_MARL.num_agents)}

    def get_time_obs(self):
        obs = np.zeros((self.num_dims_time,))
        if self.obs_time:
            if self.one_hot_time:
                obs[self.env_MARL.t % self.num_dims_time] = 1
            else:
                obs[0] = (self.env_MARL.t % self.num_dims_time) / self.num_dims_time
        return obs

    def step(self, actions):
        if self.unbounded_actions:
            actions.update((k, (np.tanh(actions[k]) + 1) / 2 * (self.u_high - self.u_low) + self.u_low) if k != -1 else
                           (k, (np.tanh(actions[k]) + 1) / 2 * (self.major_u_high - self.major_u_low) + self.major_u_low)
                           for k in actions.keys())
        xs, rewards, dones, _ = self.env_MARL.step({i: actions[i] for i in actions if i!=-1}, actions[-1])
        return self.get_observation(), rewards, dones, {}

    def render(self, mode="human"):
        pass

    def observation_space_sample(self, agent_ids: list = None):
        return self.get_observation()

    def observation_space_contains(self, x):
        return np.all([self.observation_space.contains(item) for item in x.values()])

    def action_space_sample(self, agent_ids: list = None):
        return {i: self.action_space.sample() for i in range(-1, self.env_MARL.num_agents)}

    def action_space_contains(self, x):
        return np.all([self.action_space.contains(item) for item in x.values()])
