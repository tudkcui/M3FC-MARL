import matplotlib.pyplot as plt
import numpy as np
import ot
from gym.spaces import Box
from torch.distributions import Uniform

from marllib.envs.m3fc_envs.mf_marl_env import MeanFieldMARLEnv

class CommsTwoGaussiansEnv(MeanFieldMARLEnv):
    """
    Models the UAV-based communication cell coverage problem.
    """
    def __init__(self, num_agents: int = 100, time_steps: int = 100, x_low: float = -2., x_high: float = 2.,
                 u_low: float = -1, u_high: float = 1, capacity_ratio=1., v_max=0.2, num_users: int = 100,
                 noise_std: float = 0.03, dims: int = 2, period_time: int = 50, cost_function='sqeuclidean',
                 **kwargs, ):
        self.x_low = x_low
        self.x_high = x_high
        self.noise_std = noise_std
        self.period_time = period_time
        self.distance_setting = cost_function
        self.capacity_ratio = capacity_ratio
        self.v_max = v_max
        self.num_users = num_users
        self.dims = dims

        self.nu = None
        self.move_vecs = None

        agent_observation_space = Box(x_low, x_high, shape=(dims,))
        # agent_observation_space = MultiAgentObservationSpace([Box(low=x_low, high=x_high, shape=(dims,), dtype=np.float32)
        #      for _ in range(num_agents)])

        agent_action_space = Box(u_low, u_high, shape=(dims,))
        # agent_action_space = MultiAgentActionSpace([Box(low=u_low, high=u_high, shape=(dims,), dtype=np.float32)
        #                                       for _ in range(num_agents)])
        super().__init__(agent_observation_space, agent_action_space, time_steps, num_agents=num_agents, **kwargs)

    def sample_initial_state(self):
        return Uniform(self.x_low, self.x_high).sample((self.dims,)).numpy()


    # def close(self):
    #     pass

    def reset(self):
        super().reset()
        self.sample_user_particles()
        return self.get_observation()

    def sample_user_particles(self):
        time_of_day = self.t % self.period_time
        angle = 2 * np.pi * (time_of_day / self.period_time)
        x = np.cos(angle)
        num_gaussian_one = int(self.num_users * (x + 1) // 2)
        num_gaussian_two = self.num_users - num_gaussian_one
        gaussian_samples_one = np.random.multivariate_normal([-1] + [0] * (self.dims - 1),
                                                         np.diag([0.05] * self.dims), size=(num_gaussian_one,))
        gaussian_samples_two = np.random.multivariate_normal([1] + [0] * (self.dims - 1),
                                                         np.diag([0.05] * self.dims), size=(num_gaussian_two,))
        self.nu = np.clip(np.concatenate([gaussian_samples_one, gaussian_samples_two], axis=0), self.x_low, self.x_high)

    def next_states(self, t, xs, us):
        self.move_vecs = us / (np.expand_dims((np.linalg.norm(us, axis=1) <= 1)
                                              + (np.linalg.norm(us, axis=1) > 1) * np.linalg.norm(us, axis=1), axis=1)
                               + 1e-10)
        return np.clip(xs + self.v_max * self.move_vecs
                       + np.random.normal(0, self.noise_std, size=xs.shape), self.x_low, self.x_high)

    def update(self):
        self.sample_user_particles()

    def reward(self, t, xs, us):
        dist_cost = self.find_ot_cost(xs)
        act_cost = 0.0 * np.mean(np.linalg.norm(self.move_vecs, axis=1))
        return - dist_cost - act_cost

    def find_ot_cost(self, xs):
        drone_particles = np.hstack([xs, np.ones((len(xs), 1))])
        num_extra_particles = round((self.capacity_ratio - 1) * self.num_users)
        user_particles = np.hstack([self.nu, np.ones((len(self.nu), 1))])

        if self.distance_setting == 'sqeuclidean':
            M_partial = ot.dist(drone_particles, user_particles)
            M = np.hstack([M_partial, np.zeros((drone_particles.shape[0], num_extra_particles))])
        elif self.distance_setting == 'abs':
            M_partial = np.sqrt(ot.dist(drone_particles, user_particles))
            M = np.hstack([M_partial, np.zeros((drone_particles.shape[0], num_extra_particles))])
        elif self.distance_setting == 'sqeuclidean-cutoff':
            M_partial = ot.dist(drone_particles, user_particles)
            M_partial = (M_partial > 1) * np.ones_like(M_partial) + (M_partial <= 1) * M_partial
            M = np.hstack([M_partial, np.zeros((drone_particles.shape[0], num_extra_particles))])
        elif self.distance_setting == 'abs-cutoff':
            M_partial = np.sqrt(ot.dist(drone_particles, user_particles))
            M_partial = (M_partial > 1) * np.ones_like(M_partial) + (M_partial <= 1) * M_partial
            M = np.hstack([M_partial, np.zeros((drone_particles.shape[0], num_extra_particles))])
        else:
            raise NotImplementedError

        return ot.emd2(np.ones((self.num_agents,)) / self.num_agents,
                       np.ones((self.num_users + num_extra_particles,)) / (self.num_users + num_extra_particles), M)

    def render(self, mode='human'):
        plt.style.use('ggplot')
        plt.close('all')
        plt.scatter(self.xs[:, 0], self.xs[:, 1], color='red')  # 2D case
        plt.scatter(self.nu[:, 0], self.nu[:, 1], color='blue')
        plt.xlim([self.x_low, self.x_high])
        plt.ylim([self.x_low, self.x_high])
        plt.show()
        plt.pause(0.1)


