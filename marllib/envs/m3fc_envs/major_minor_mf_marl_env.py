from abc import ABC, abstractmethod

import numpy as np


class MajorMinorMARLEnv(ABC):
    """
    Models a major-minor mean-field MARL problem in discrete time, i.e. MDPs with additional population components.
    """

    def __init__(self, minor_observation_space, minor_action_space, major_observation_space, major_action_space,
                 time_steps, num_agents,
                 vectorized=True, scale_rewards=False, **kwargs):
        self.minor_observation_space = minor_observation_space
        self.minor_action_space = minor_action_space
        self.major_observation_space = major_observation_space
        self.major_action_space = major_action_space
        self.time_steps = time_steps
        self.num_agents = num_agents
        self.vectorized = vectorized
        self.scale_rewards = scale_rewards
        self._agent_ids = list(range(self.num_agents))

        super().__init__()

        self.xs = None
        self.y = None
        self.t = None

    def reset(self):
        self.t = 0
        self.xs = np.array([self.sample_initial_minor_state() for _ in range(self.num_agents)])
        self.y = self.sample_initial_major_state()
        return self.get_observation()

    def get_observation(self):
        if self.vectorized:
            return self.xs, self.y
        else:
            out = {i: self.xs[i] for i in range(self.num_agents)}
            return out, self.y

    def action_space_sample(self, agent_ids: list = None):
        out = {i: self.minor_action_space.sample() for i in range(self.num_agents)}
        out['major'] = self.major_action_space.sample()
        return out

    @abstractmethod
    def sample_initial_minor_state(self):
        pass

    @abstractmethod
    def sample_initial_major_state(self):
        pass

    def step(self, action_minor, action_major):
        if not self.vectorized or type(action_minor) == dict:
            action_minor = np.array(list(a[1] for a in action_minor.items()))
        next_xs, next_y = self.next_states(self.t, self.xs, self.y, action_minor, action_major)
        reward = self.reward(self.t, self.xs, self.y, action_minor, action_major)

        self.t += 1
        self.xs = next_xs
        self.y = next_y
        self.update()

        if self.vectorized:
            return self.get_observation(), reward, {'__all__': self.t >= self.time_steps}, {}
        else:
            return self.get_observation(), {i: reward * self.num_agents if self.scale_rewards else reward
                                            for i in range(-1, self.num_agents)}, {'__all__': self.t >= self.time_steps}, {}

    """
    Note that for fast execution, we vectorize and use the states and actions of all agents directly. 
     The implementing class makes sure that the next states and reward function follow the MFC model assumptions. """
    @abstractmethod
    def next_states(self, t, xs, y, us, u_major):
        pass  # sample new states for all agents

    @abstractmethod
    def reward(self, t, xs, y, us, u_major):
        pass  # sample reward defined on the state-action mean-field

    def update(self):
        pass
