import numpy as np
from gym.spaces import Box, Tuple
from ray.rllib import MultiAgentEnv

from marllib.envs.m3fc_envs.major_minor_mf_marl_env import MajorMinorMARLEnv


class MajorMinorMARLObsWrapperDiscretePureTupleSeparate(MultiAgentEnv):
    """
    Wraps a mean-field MARL problem in discrete time with extra observations.
    """
    def __init__(self, env_MARL: MajorMinorMARLEnv, unbounded_actions=False,
                 obs_time=0, periodic_time=False, **kwargs, ):
        super().__init__()
        self.env_MARL = env_MARL
        self.obs_time = obs_time >= 1
        self.one_hot_time = obs_time >= 2
        self.periodic_time = periodic_time
        self.unbounded_actions = unbounded_actions

        self.num_dims_time = 0 if not self.obs_time \
            else 1 if not self.one_hot_time \
            else self.env_MARL.period_time if self.periodic_time \
            else self.env_MARL.time_steps

        self._agent_ids = list(range(-1, self.env_MARL.num_agents))

        self.num_states = self.env_MARL.minor_observation_space[0].n
        self.x_dims = len(self.env_MARL.minor_observation_space)
        self.num_obs_layers = 1 + self.num_dims_time
        self.observation_space = Tuple([
            self.env_MARL.minor_observation_space,
            self.env_MARL.major_observation_space,
        ] + [Box(0, 1, shape=(self.num_states ** self.x_dims * self.num_obs_layers, ), dtype=np.float64),])
        self.action_space = self.env_MARL.minor_action_space

    def get_histogram_from(self, xs, normalizer=None):
        obs = np.zeros((self.num_states,) * self.x_dims + (1,))
        for idx in xs:
            obs[idx] += 1
        obs /= self.env_MARL.num_agents if normalizer is None else normalizer
        return obs

    def reset(self):
        self.env_MARL.reset()
        return self.get_observation()

    def get_observation(self):
        xs, y = self.env_MARL.get_observation()
        obs_agents = self.get_histogram_from(self.env_MARL.xs)
        if self.obs_time:
            obs_time = self.get_time_obs()
            return {i: (xs[i], y, np.concatenate([obs_agents, obs_time], axis=-1).flatten()) if i != -1 else
                    (0 * xs[0], y, np.concatenate([obs_agents, obs_time], axis=-1).flatten())
                    for i in range(-1, self.env_MARL.num_agents)}
        else:
            return {i: (xs[i], y, np.concatenate([obs_agents], axis=-1).flatten()) if i != -1 else
                    (0 * xs[0], y, np.concatenate([obs_agents], axis=-1).flatten())
                    for i in range(-1, self.env_MARL.num_agents)}

    def get_time_obs(self):
        obs = np.zeros((self.num_dims_time,))
        if self.obs_time:
            if self.one_hot_time:
                obs[self.env_MARL.t % self.num_dims_time] = 1
            else:
                obs[0] = (self.env_MARL.t % self.num_dims_time) / self.num_dims_time
        return obs

    def step(self, actions):
        xs, rewards, dones, _ = self.env_MARL.step({i: actions[i] for i in actions if i!=-1}, actions[-1])
        return self.get_observation(), rewards, dones, {}

    def render(self, mode="human"):
        pass

    def observation_space_sample(self, agent_ids: list = None):
        return self.get_observation()

    def observation_space_contains(self, x):
        return np.all([self.observation_space.contains(item) for item in x.values()])

    def action_space_sample(self, agent_ids: list = None):
        return {i: self.action_space.sample() for i in range(-1, self.env_MARL.num_agents)}

    def action_space_contains(self, x):
        return np.all([self.action_space.contains(item) for item in x.values()])
