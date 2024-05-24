import numpy as np
import torch
from gym.spaces import Discrete, Tuple
from torch.distributions import Categorical

from marllib.envs.m3fc_envs.major_minor_mf_marl_env import MajorMinorMARLEnv


class MajorMinorBeachTupleEnv(MajorMinorMARLEnv):
    """
    Models the Beach problem around a central beach bar agent, using tuple states
    """

    def __init__(self, num_agents: int = 100, time_steps: int = 200, grid_length: int = 5, dims: int = 1,
                 cost_distance_minor: float = 1.0, cost_distance_major: float = 0.3,
                 cost_crowd: float = 0.06, prob_move_center: float = 0.2, moves_center: int = 1, **kwargs, ):
        self.dims = dims
        self.grid_length = grid_length
        self.num_states = grid_length ** dims
        self.cost_distance_minor = cost_distance_minor
        self.cost_distance_major = cost_distance_major
        self.cost_crowd = cost_crowd
        self.prob_move_center = prob_move_center
        self.moves_center = moves_center

        self.curr_target_major = 0

        agent_observation_space = Tuple([Discrete(self.grid_length)] * dims)
        agent_action_space = Discrete(2 * dims + 1)
        major_observation_space = Tuple([Tuple([Discrete(self.grid_length)] * dims),
                                         Tuple([Discrete(self.grid_length)] * dims)])
        super().__init__(agent_observation_space, agent_action_space, major_observation_space, agent_action_space,
                         time_steps, num_agents=num_agents, **kwargs)

    def sample_initial_major_state(self):
        return tuple([Categorical(probs=torch.tensor([1/self.grid_length] * self.grid_length)).sample().numpy().item()
                      for _ in range(self.dims)])

    def sample_initial_minor_state(self):
        return tuple([Categorical(probs=torch.tensor([1/self.grid_length] * self.grid_length)).sample().numpy().item()
                      for _ in range(self.dims)])

    def get_observation(self):
        if self.vectorized:
            return self.xs, (self.y, self.curr_target_major)
        else:
            out = {i: self.xs[i] for i in range(self.num_agents)}
            return out, (self.y, self.curr_target_major)

    def reset(self):
        super().reset()
        self.curr_target_major = np.copy(self.y)
        self.resample_targets()
        return self.get_observation()

    def update(self):
        self.resample_targets()

    def resample_targets(self):
        if np.random.rand() < self.prob_move_center:
            for _ in range(self.moves_center):
                position = np.expand_dims(self.curr_target_major, axis=0)
                random_u = np.expand_dims(self.minor_action_space.sample(), axis=0)
                dir = sum([np.tile([[0] * d + [-1] + [0] * (self.dims - d - 1)], (1, 1)) * np.expand_dims(random_u == (2*d+1), axis=1)
                            + np.tile([[0] * d + [1] + [0] * (self.dims - d - 1)], (1, 1)) * np.expand_dims(random_u == (2*d+2), axis=1)
                            for d in range(self.dims)])
                self.curr_target_major = np.squeeze((position + dir) % self.grid_length, axis=0)

    def to_xs(self, positions):
        return np.sum(np.expand_dims([self.grid_length ** d for d in range(self.dims)], axis=0) * positions, axis=1)

    def from_xs(self, xs):
        return np.stack([(xs // (self.grid_length ** d)) % self.grid_length for d in range(self.dims)]).transpose()

    def next_states(self, t, xs, y, us, u_major):
        positions = np.copy(xs)
        dirs = sum([np.tile([[0] * d + [-1] + [0] * (self.dims - d - 1)], (self.num_agents, 1)) * np.expand_dims(us == (2*d+1), axis=1)
                    + np.tile([[0] * d + [1] + [0] * (self.dims - d - 1)], (self.num_agents, 1)) * np.expand_dims(us == (2*d+2), axis=1)
                    for d in range(self.dims)])
        next_xs = (positions + dirs) % self.grid_length

        position_y = np.expand_dims(y, axis=0)
        u_y = np.expand_dims(u_major, axis=0)
        dirs = sum([np.tile([[0] * d + [-1] + [0] * (self.dims - d - 1)], (1, 1)) * np.expand_dims(u_y == (2*d+1), axis=1)
                    + np.tile([[0] * d + [1] + [0] * (self.dims - d - 1)], (1, 1)) * np.expand_dims(u_y == (2*d+2), axis=1)
                    for d in range(self.dims)])
        next_y = np.squeeze((position_y + dirs) % self.grid_length, axis=0)

        return next_xs, next_y

    def reward(self, t, xs, y, us, u_major):
        pos_xs = xs
        pos_y = y
        pos_targ = self.curr_target_major

        dists_minors = np.array([np.abs(pos_xs - pos_y)]
                                 + [np.abs(pos_xs - (pos_y + np.array([0] * d
                                                                   + [self.grid_length]
                                                                   + [0] * (self.dims - d - 1))))
                                    for d in range(self.dims)]
                                 + [np.abs(pos_xs - (pos_y + np.array([0] * d
                                                                   + [-self.grid_length]
                                                                   + [0] * (self.dims - d - 1))))
                                    for d in range(self.dims)]
                                 ).min(axis=0).sum(axis=1)  # toroidal L1 dist
        dist_cost_minor = np.mean(self.cost_distance_minor * dists_minors)

        dist_major = np.array([np.abs(np.expand_dims(pos_targ, axis=0) - pos_y)]
                                 + [np.abs(np.expand_dims(pos_targ, axis=0) - (pos_y + np.array([0] * d
                                                                   + [self.grid_length]
                                                                   + [0] * (self.dims - d - 1))))
                                    for d in range(self.dims)]
                                 + [np.abs(np.expand_dims(pos_targ, axis=0) - (pos_y + np.array([0] * d
                                                                   + [-self.grid_length]
                                                                   + [0] * (self.dims - d - 1))))
                                    for d in range(self.dims)]
                                 ).min(axis=0).sum(axis=1)  # toroidal L1 dist
        dist_cost_major = np.mean(self.cost_distance_major * dist_major)

        crowd_cost = np.mean(self.cost_crowd *
                             (np.expand_dims(self.to_xs(xs), axis=0) == np.expand_dims(self.to_xs(xs), axis=1))) * self.num_states

        return - dist_cost_minor - dist_cost_major - crowd_cost
