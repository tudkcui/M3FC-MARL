import numpy as np
from gym.spaces import Box, Tuple

from marllib.envs.m3fc_envs.major_minor_mf_marl_env import MajorMinorMARLEnv
from marllib.envs.m3fc_envs.wrappers.major_minor_discrete import StateDiscretizedMajorMinorMFCEnv


class DiscreteGaussianMajorMinorMFCEnv(StateDiscretizedMajorMinorMFCEnv):

    """
    Wraps a spatial MFMARL problem with continuous state and continuous action space.
    Describes an MFMDP through discretized states and rescaled diagonal Gaussian actions, passing through major action.
    """
    def __init__(self, env_MARL: MajorMinorMARLEnv, x_bins_per_dim: int = 5, u_bins_per_dim: int = 3,
                 unbounded_actions=True, **kwargs):
        super().__init__(env_MARL, x_bins_per_dim, **kwargs)

        self.unbounded_actions = unbounded_actions
        self.true_action_shape = (x_bins_per_dim,) * self.x_dims + (self.u_dims, 2,)  # workaround rllib

        if self.unbounded_actions:
            self.action_space = Tuple([Box(-np.inf, np.inf, shape=(np.prod(self.true_action_shape).item(),)),
                                       env_MARL.major_action_space])
        else:
            self.action_space = Tuple([Box(-1, 1, shape=(np.prod(self.true_action_shape).item(),)),
                                       env_MARL.major_action_space])

    def sample_actions(self, action):
        action_minor = action[0]
        action_major = action[1]

        # if self.unbounded_actions:
        action_minor = np.tanh(action_minor)

        action_minor = np.reshape(action_minor, self.true_action_shape)
        x_idxs = self.get_bin_index_from_position(self.env_MARL.xs)
        mean = (action_minor[tuple(x_idxs)][..., 0] + 1) / 2 * (self.u_high - self.u_low) + self.u_low
        cov = (action_minor[tuple(x_idxs)][..., 1] + 1 + 1e-10) / 4
        actions_minor = np.clip(np.random.normal(mean, cov), self.u_low, self.u_high)

        return actions_minor, action_major
