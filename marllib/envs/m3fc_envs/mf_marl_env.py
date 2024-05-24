from abc import ABC, abstractmethod

import numpy as np


class ChainedTupleDistribution:
    def __init__(self, dist1, dist2):
        self.dist1 = dist1
        self.dist2 = dist2


class MeanFieldMARLEnv(ABC):
    """
    Models a pure mean-field MARL problem in discrete time.
    """

    def __init__(self, agent_observation_space, agent_action_space, time_steps, num_agents,
                 vectorized=True, scale_rewards=False, **kwargs):
        self.observation_space = agent_observation_space
        self.action_space = agent_action_space
        self.time_steps = time_steps
        self.num_agents = num_agents
        self.vectorized = vectorized
        self.scale_rewards = scale_rewards
        self._agent_ids = list(range(self.num_agents))

        super().__init__()

        self.xs = None
        self.t = None

    def reset(self):
        self.t = 0
        self.xs = np.array([self.sample_initial_state() for _ in range(self.num_agents)])
        return self.get_observation()

    def get_observation(self):
        if self.vectorized:
            return self.xs
        else:
            return {i: self.xs[i] for i in range(self.num_agents)}

    def action_space_sample(self, agent_ids: list = None):
        return {i: self.action_space.sample() for i in range(self.num_agents)}

    @abstractmethod
    def sample_initial_state(self):
        pass

    def step(self, actions):
        if isinstance(actions, dict):
            actions = np.array(list(a[1] for a in actions.items()))
        # if not self.vectorized:
        #     actions = np.array(list(a[1] for a in actions.items()))
        next_xs = self.next_states(self.t, self.xs, actions)
        reward = self.reward(self.t, self.xs, actions)

        self.t += 1
        self.xs = next_xs
        self.update()

        if self.vectorized:
            return self.get_observation(), reward, {'__all__': self.t >= self.time_steps}, {}
        else:
            return self.get_observation(), {i: reward * self.num_agents if self.scale_rewards else reward
                                            for i in range(self.num_agents)}, {'__all__': self.t >= self.time_steps}, {}

    """
    Note that for fast execution, we vectorize and use the states and actions of all agents directly. 
     The implementing class makes sure that the next states and reward function follow the MFC model assumptions. """
    @abstractmethod
    def next_states(self, t, xs, us):
        pass  # sample new states for all agents

    @abstractmethod
    def reward(self, t, xs, us):
        pass  # sample reward defined on the state-action mean-field

    def update(self):
        pass


