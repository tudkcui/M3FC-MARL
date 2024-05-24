import matplotlib.pyplot as plt
import numpy as np
import ot
from gym.spaces import Box
from torch.distributions import Uniform

from marllib.envs.m3fc_envs.major_minor_mf_marl_env import MajorMinorMARLEnv


class MajorMinorFormationEnv(MajorMinorMARLEnv):
    """
    Models the formation flight problem around a central agent, trading off between formation and moving target.
    """

    def __init__(self, num_agents: int = 100, time_steps: int = 100, x_low: float = -2., x_high: float = 2.,
                 u_low: float = -1, u_high: float = 1, v_max=0.2, noise_std: float = 0.0, dims: int = 2,
                 noise_std_target: float = 0.02, cost_function='sqeuclidean',
                 lengthscale_target_gaussian: float = 0.3, num_users: int = 300, **kwargs, ):
        self.x_low = x_low
        self.x_high = x_high
        self.noise_std = noise_std
        self.noise_std_target = noise_std_target
        self.lengthscale_target_gaussian = lengthscale_target_gaussian
        self.distance_setting = cost_function
        self.v_max = v_max
        self.dims = dims
        self.num_users = num_users

        self.nu = None
        self.curr_target_major = np.zeros(self.dims)

        agent_observation_space = Box(x_low, x_high, shape=(dims,), dtype=np.float32)
        agent_action_space = Box(u_low, u_high, shape=(dims,), dtype=np.float32)
        major_observation_space = Box(x_low, x_high, shape=(dims * 2,), dtype=np.float32)
        super().__init__(agent_observation_space, agent_action_space, major_observation_space, agent_action_space,
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
        self.curr_target_major = np.clip(np.random.multivariate_normal(self.curr_target_major * 0.95, self.noise_std_target * np.diag(np.ones(self.dims))), self.x_low, self.x_high)
        gaussian_samples = np.random.multivariate_normal(self.y, self.lengthscale_target_gaussian * np.diag(np.ones(self.dims)), size=(self.num_users,))
        self.nu = np.clip(gaussian_samples, self.x_low, self.x_high)

    def next_states(self, t, xs, y, us, u_major):
        move_vecs = us / np.expand_dims((np.linalg.norm(us, axis=1) <= 1)
                                              + (np.linalg.norm(us, axis=1) > 1) * np.linalg.norm(us, axis=1), axis=1)
        move_major = u_major if np.linalg.norm(u_major) <= 1 else u_major / np.linalg.norm(u_major)
        new_xs = np.clip(xs + self.v_max * move_vecs
                       + np.random.normal(0, self.noise_std, size=xs.shape), self.x_low, self.x_high)
        new_y = np.clip(y + self.v_max * move_major
                       + np.random.normal(0, self.noise_std, size=y.shape), self.x_low, self.x_high)
        return new_xs, new_y

    def reward(self, t, xs, y, us, u_major):
        agents = np.hstack([xs, np.ones((len(xs), 1))])
        targets = np.hstack([self.nu, np.ones((len(self.nu), 1))])

        if self.distance_setting == 'sqeuclidean':
            M = ot.dist(agents, targets)
        elif self.distance_setting == 'abs':
            M = np.sqrt(ot.dist(agents, targets))
        else:
             M = ot.dist(agents, targets, metric=self.distance_setting)

        minor_reward = - ot.emd2(np.ones((self.num_agents,)) / self.num_agents,
                                 np.ones((self.num_users,)) / self.num_users, M)
        major_reward = - np.linalg.norm(y - self.curr_target_major)

        return minor_reward + major_reward

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
        plt.scatter(self.nu[:, 0], self.nu[:, 1], color='blue')
        plt.xlim([self.x_low, self.x_high])
        plt.ylim([self.x_low, self.x_high])

        plt.show()
        plt.pause(0.05)
