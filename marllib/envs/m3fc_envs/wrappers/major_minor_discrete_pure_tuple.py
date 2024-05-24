import numpy as np
import torch
import torch.nn.functional as F
from gym.spaces import Box, Tuple

from marllib.envs.m3fc_envs.major_minor_mf_marl_env import MajorMinorMARLEnv
from marllib.envs.m3fc_envs.wrappers.major_minor_mfc import MajorMinorMFCEnv


class PureDiscreteTupleMajorMinorMFCEnv(MajorMinorMFCEnv):

    """
    Wraps a MM-MFC with discrete state and discrete action space.
    """
    def __init__(self, env_MARL: MajorMinorMARLEnv, unbounded_actions=True, **kwargs):
        super().__init__(env_MARL, **kwargs)

        self.num_states = self.env_MARL.minor_observation_space[0].n
        self.x_dims = len(self.env_MARL.minor_observation_space)
        self.num_obs_layers = 1 + self.num_dims_time
        self.observation_space = Tuple([Box(0, 1, shape=(self.num_states,) * self.x_dims + (self.num_obs_layers,), dtype=np.float64),
                                        env_MARL.major_observation_space])

        self.unbounded_actions = unbounded_actions
        self.u_bins_per_dim = self.env_MARL.minor_action_space.n
        self.true_action_shape = (self.num_states,) * self.x_dims + (self.u_bins_per_dim,)
        self.u_bin_centers = np.array(range(self.u_bins_per_dim))

        if self.unbounded_actions:
            self.action_space = Tuple([Box(-np.inf, np.inf, shape=(np.prod(self.true_action_shape).item(),)),
                                       env_MARL.major_action_space])
        else:
            self.action_space = Tuple([Box(-1, 1, shape=(np.prod(self.true_action_shape).item(),)),
                                       env_MARL.major_action_space])

    def get_observation(self):
        xs, y = self.env_MARL.get_observation()
        obs_agents = self.get_histogram_from(xs)
        if self.obs_time:
            obs_time = np.tile(self.get_time_obs(), (self.num_states,) * self.x_dims + (1,))
            return np.concatenate([obs_agents, obs_time], axis=-1), y
        else:
            return obs_agents, y

    def get_histogram_from(self, xs, normalizer=None):
        obs = np.zeros((self.num_states,) * self.x_dims + (1,))
        for idx in xs:
            obs[idx] += 1
        obs /= self.env_MARL.num_agents if normalizer is None else normalizer
        return obs

    def sample_actions(self, action):
        action_minor = action[0]
        action_major = action[1]
        if self.unbounded_actions:  # for a2c use this
            h = F.softmax(torch.tensor(np.reshape(action_minor, self.true_action_shape)), dim=-1).numpy()
        else:
            action_minor = np.reshape(action_minor, self.true_action_shape) + 1 + 1e-10
            h = action_minor / np.sum(action_minor, axis=-1, keepdims=True)

        """ Compute MFC action probs """
        cum_h = np.cumsum(h, axis=-1)
        cum_ps = np.concatenate([np.zeros((self.env_MARL.num_agents, 1)), cum_h[tuple(self.env_MARL.xs.transpose())]], axis=-1)

        """ Compute actions based on MFC """
        actions_minor = np.zeros((self.env_MARL.num_agents,), dtype=np.int64)
        idxs = np.tile(np.array(list(range(self.u_bins_per_dim)), dtype=int), (self.env_MARL.num_agents, 1))
        uniform_samples = np.random.uniform(0, 1, size=self.env_MARL.num_agents)
        actions_one_hot = np.zeros((self.env_MARL.num_agents, self.u_bins_per_dim), dtype=int)
        for idx in range(self.u_bins_per_dim):
            actions_one_hot[:, idx] = np.logical_and(uniform_samples >= cum_ps[:, idx], uniform_samples < cum_ps[:, idx+1])
        actions_idxs = np.sum(actions_one_hot * idxs, axis=1)
        actions_minor[:] = self.u_bin_centers[actions_idxs]
        return actions_minor, action_major
