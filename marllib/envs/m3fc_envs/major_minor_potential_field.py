import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box
from torch.distributions import Uniform

from marllib.envs.m3fc_envs.major_minor_mf_marl_env import MajorMinorMARLEnv


class MajorMinorPotentialFieldEnv(MajorMinorMARLEnv):
    """
    Models the formation flight problem around a central agent, trading off between formation and moving target.
    """

    def __init__(self, num_agents: int = 100, time_steps: int = 100, x_low: float = -2., x_high: float = 2.,
                 u_low: float = -1, u_high: float = 1, v_max=0.2, noise_std: float = 0.0, dims: int = 2,
                 v_max_major: float = 0., range: float = 1.0, num_users: int = 300,
                 noise_std_target: float = 0.005,
                 **kwargs, ):
        self.x_low = x_low
        self.x_high = x_high
        self.noise_std = noise_std
        self.noise_std_target = noise_std_target
        self.v_max = v_max
        self.v_max_major = v_max_major
        self.dims = dims
        self.num_users = num_users
        self.range = range

        self.curr_target_major = np.zeros(self.dims)

        agent_observation_space = Box(x_low, x_high, shape=(dims,), dtype=np.float64)
        agent_action_space = Box(u_low, u_high, shape=(dims,), dtype=np.float64)
        major_observation_space = Box(x_low, x_high, shape=(dims * 2,), dtype=np.float64)
        major_action_space = Box(u_low, u_high, shape=(dims,), dtype=np.float64)
        super().__init__(agent_observation_space, agent_action_space, major_observation_space, major_action_space,
                         time_steps, num_agents=num_agents, **kwargs)

    def sample_initial_major_state(self):
        return Uniform(self.x_low, self.x_high).sample((self.dims,)).numpy()

    def sample_initial_minor_state(self):
        return Uniform(self.x_low, self.x_high).sample((self.dims,)).numpy()

    def get_observation(self):
        if self.vectorized:
            return self.xs, np.hstack([self.y, self.curr_target_major])
        else:
            out = {i: self.xs[i] for i in range(self.num_agents)}
            return out, np.hstack([self.y, self.curr_target_major])

    def reset(self):
        super().reset()
        self.curr_target_major = np.zeros(self.dims)
        self.resample_targets()
        return self.get_observation()

    def update(self):
        self.resample_targets()

    def resample_targets(self):
        self.curr_target_major = np.clip(np.random.multivariate_normal(self.curr_target_major * 0.99, self.noise_std_target * np.diag(np.ones(self.dims))), self.x_low, self.x_high)

    def next_states(self, t, xs, y, us, u_major):
        move_vecs = us / np.expand_dims((np.linalg.norm(us, axis=1) <= 1)
                                              + (np.linalg.norm(us, axis=1) > 1) * np.linalg.norm(us, axis=1), axis=1)
        move_major = u_major if np.linalg.norm(u_major) <= 1 else u_major / np.linalg.norm(u_major)
        shifts = [(self.x_high - self.x_low, self.x_high - self.x_low),
                  (self.x_low - self.x_high, self.x_high - self.x_low),
                  (self.x_high - self.x_low, self.x_low - self.x_high),
                  (self.x_low - self.x_high, self.x_low - self.x_high)] if self.dims == 2 else \
                [(self.x_high - self.x_low), (self.x_low - self.x_high)]
        vecs_xs_to_y_all = np.array([np.expand_dims(y, axis=0) - xs]
                                + [np.expand_dims(y + shift, axis=0) - xs for shift in shifts])
        vecs_xs_to_y = vecs_xs_to_y_all[np.linalg.norm(vecs_xs_to_y_all, axis=2).argmin(axis=0), np.arange(self.num_agents)]
        dist_xs_to_y = np.linalg.norm(vecs_xs_to_y, axis=1, keepdims=True)
        new_xs = (xs + self.v_max * move_vecs + np.random.normal(0, self.noise_std, size=xs.shape)
                  - self.x_low) % (self.x_high - self.x_low) + self.x_low
        new_y = (y + self.v_max_major * move_major + np.random.normal(0, self.noise_std, size=y.shape)
                 + self.v_max / 2 * np.mean((dist_xs_to_y < self.range)
                                            * np.maximum(0, self.range - dist_xs_to_y)
                                            * (vecs_xs_to_y) / (dist_xs_to_y + 1e-10), axis=0)
                 - self.x_low) % (self.x_high - self.x_low) + self.x_low
        return new_xs, new_y

    def reward(self, t, xs, y, us, u_major):
        return - np.linalg.norm(y - self.curr_target_major)

    def render(self, mode='human'):
        plt.style.use('ggplot')
        plt.close('all')

        plt.subplot(1, 2, 1)
        plt.scatter(self.y[0], self.y[1], color='red', s=20)  # 2D case
        plt.scatter(self.curr_target_major[0], self.curr_target_major[1], color='orange', s=20)
        plt.xlim([self.x_low, self.x_high])
        plt.ylim([self.x_low, self.x_high])

        plt.subplot(1, 2, 2)
        plt.scatter(self.xs[:, 0], self.xs[:, 1], color='black')
        plt.xlim([self.x_low, self.x_high])
        plt.ylim([self.x_low, self.x_high])

        plt.show()
        plt.pause(0.05)
