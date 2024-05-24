import numpy as np
from gym.spaces import Box

from marllib.envs.m3fc_envs.mf_marl_env import MeanFieldMARLEnv
from marllib.envs.m3fc_envs.wrappers.discrete import StateDiscretizedMFCEnv


class DiscreteGaussianMFCEnv(StateDiscretizedMFCEnv):

    """
    Wraps a spatial MFMARL problem with continuous state and continuous action space.
    Describes an MFMDP through discretized states and rescaled diagonal Gaussian actions.
    """
    def __init__(self, env_MARL: MeanFieldMARLEnv, x_bins_per_dim: int = 5, unbounded_actions=True, **kwargs):
        super().__init__(env_MARL, x_bins_per_dim, **kwargs)

        self.unbounded_actions = unbounded_actions
        self.true_action_shape = (x_bins_per_dim,) * self.x_dims + (self.u_dims, 2,)  # workaround rllib

        if self.unbounded_actions:
            self.action_space = Box(-np.inf, np.inf, shape=(np.prod(self.true_action_shape).item(),))
        else:
            self.action_space = Box(-1, 1, shape=(np.prod(self.true_action_shape).item(),))

    def sample_actions(self, action):
        # if self.unbounded_actions:
        action = np.tanh(action)

        action = np.reshape(action, self.true_action_shape)
        x_idxs = self.get_bin_index_from_position(self.env_MARL.xs)
        mean = (action[tuple(x_idxs)][..., 0] + 1) / 2 * (self.u_high - self.u_low) + self.u_low
        cov = (action[tuple(x_idxs)][..., 1] + 1 + 1e-10) / 4
        return np.clip(np.random.normal(mean, cov), self.u_low, self.u_high)
