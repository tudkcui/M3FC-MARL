import matplotlib.pyplot as plt
import numpy as np
import ot
from gym.spaces import Box
from torch.distributions import Uniform

from marllib.envs.m3fc_envs.major_minor_mf_marl_env import MajorMinorMARLEnv


class MajorMinorForagingEnv(MajorMinorMARLEnv):
    """
    Models the foraging problem around a slow central agent (truck), collecting stuffs.
    """

    def __init__(self, num_agents: int = 100, time_steps: int = 200, x_low: float = -2., x_high: float = 2.,
                 u_low: float = -1, u_high: float = 1, v_max_major=0.1, v_max_minor=0.3, noise_std: float = 0.0,
                 dims: int = 2, range_task: float = 0.5, **kwargs, ):
        self.x_low = x_low
        self.x_high = x_high
        self.noise_std = noise_std
        self.v_max_major = v_max_major
        self.v_max_minor = v_max_minor
        self.dims = dims
        self.range_task = range_task

        self.nu = np.array([])
        self.remaining_task_lengths = []
        self.move_vecs = None
        self.last_reward = 0
        self.num_users = 5
        self.curr_target_major = np.zeros(self.dims)

        agent_observation_space = Box(np.array([x_low] * dims + [0]), np.array([x_high] * dims + [1]), shape=(dims + 1,), dtype=np.float64)
        agent_action_space = Box(u_low, u_high, shape=(dims,))
        major_observation_space = Box(x_low, x_high, shape=(dims,), dtype=np.float64)
        super().__init__(agent_observation_space, agent_action_space, major_observation_space, agent_action_space,
                         time_steps, num_agents=num_agents, **kwargs)

    def sample_initial_major_state(self):
        return np.clip(Uniform(self.x_low, self.x_high).sample((self.dims,)).numpy(),
                       [self.x_low, self.x_low], [self.x_high, self.x_low / 2])

    def sample_initial_minor_state(self):
        return np.concatenate([Uniform(self.x_low, self.x_high).sample((self.dims,)).numpy(), [0]])

    def reset(self):
        super().reset()
        self.nu = np.array([])
        self.remaining_task_lengths = []
        self.move_vecs = None
        self.last_reward = 0
        self.num_users = 5
        self.curr_target_major = np.zeros(self.dims)
        return self.get_observation()

    def next_states(self, t, xs, y, us, u_major):
        self.last_reward = 0

        move_vecs = us / np.expand_dims((np.linalg.norm(us, axis=1) <= 1)
                                              + (np.linalg.norm(us, axis=1) > 1) * np.linalg.norm(us, axis=1), axis=1)
        move_major = u_major if np.linalg.norm(u_major) <= 1 else u_major / np.linalg.norm(u_major)
        new_xs = np.clip(xs + self.v_max_minor * np.hstack([move_vecs, np.zeros((self.num_agents, 1))])
                       + np.random.normal(0, self.noise_std, size=xs.shape), self.x_low, self.x_high)
        new_y = np.clip(y + self.v_max_major * move_major
                       + np.random.normal(0, self.noise_std, size=y.shape),
                        [self.x_low, self.x_low], [self.x_high, self.x_low / 2])

        new_nu = []
        new_remaining_task_lengths = []

        if len(self.nu) > 0:
            """ Process existing targets """
            M = np.sqrt(ot.dist(xs[:, :self.dims], self.nu[:, :self.dims]))
            for target, task_length, dists in zip(self.nu, self.remaining_task_lengths, M.transpose()):
                weights = (self.range_task - dists) / self.range_task * (dists < self.range_task)
                processed_length = min(0.1, np.mean(weights))
                if task_length > processed_length:
                    new_nu.append(target)
                    new_remaining_task_lengths.append(task_length - processed_length)
                else:
                    processed_length = task_length

                """ Give resources to agents, any overflow is wasted """
                forage = processed_length * weights / (np.sum(weights) + 1e-10) * self.num_agents
                foraged_amount = np.clip(new_xs[:, self.dims] + forage, 0, 1) - new_xs[:, self.dims]  # Excess is wasted
                new_xs[:, self.dims] = np.clip(new_xs[:, self.dims] + forage, 0, 1)
                self.last_reward += np.mean(foraged_amount)

        """ Drop off minor resources at major agent """
        dropped_off_amount = new_xs[:, self.dims] - ((np.linalg.norm(new_xs[:, :self.dims] - np.expand_dims(self.y, axis=0), axis=-1) > self.range_task) * new_xs[:, self.dims])
        new_xs[:, self.dims] = (np.linalg.norm(new_xs[:, :self.dims] - np.expand_dims(self.y, axis=0), axis=-1) > self.range_task) * new_xs[:, self.dims]
        self.last_reward += np.mean(dropped_off_amount)

        """ Sample new targets """
        num_new_targets = min(np.random.poisson(0.2), self.num_users - len(new_nu))
        for _ in range(num_new_targets):
            target = np.concatenate([np.random.uniform(self.x_low, self.x_high, (self.dims,)), [0]], axis=0)
            length = np.random.uniform(0.5, 1.5)
            new_nu.append(target)
            new_remaining_task_lengths.append(length)

        self.nu = np.array(new_nu)
        self.remaining_task_lengths = new_remaining_task_lengths

        return new_xs, new_y

    # Use results from last next states call
    def reward(self, t, xs, y, us, u_major):
        return 100 * self.last_reward

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
