# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
An example of integrating new tasks into MARLLib
About ma-gym: https://github.com/koulanurag/ma-gym
doc: https://github.com/koulanurag/ma-gym/wiki

Learn how to transform the environment to be compatible with MARLlib:
please refer to the paper: https://arxiv.org/abs/2210.13708

Install ma-gym before use
"""
import numpy as np
from gym.wrappers import FlattenObservation
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gym.spaces import Dict as GymDict, Box
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
import time
from gym.spaces import flatten_space

from marllib.envs.m3fc_envs import args_parser
import multiprocessing
import yaml

import os

map_name = "exp1"
yaml_config_file_path = os.path.join(os.path.dirname(__file__),
                                     "../marllib/envs/base_env/config/{}.yaml".format(map_name))
# register all scenario with env class
REGISTRY = {}
# provide detailed information of each scenario
# mostly for policy sharing
policy_mapping_dict = {
    map_name: {
        "description": "one team cooperate",
        "team_prefix": ("agent_",),
        "all_agents_one_policy": True,
        "one_agent_one_policy": False,
    },
}


# must inherited from MultiAgentEnv class
class RLlibMAGym(MultiAgentEnv):

    def __init__(self, env_config):
        map = env_config["map_name"]
        env_config.pop("map_name", None)
        env_config_file_path = yaml_config_file_path
        with open(env_config_file_path, "r") as f:
            env_config_dict = yaml.load(f, Loader=yaml.FullLoader)
            f.close()
        env_config_update = args_parser.generate_config(env_config_dict)

        self.env = FlattenObservation(
            env_config_update['solver'](env_config_update['game'](**env_config_update), **env_config_update))

        self.observation_space = GymDict({"obs": flatten_space(self.env.observation_space)})

        if self.env.num_agents == 1:
            self.action_space = self.env.action_space

        self.agents = ["agent_{}".format(i) for i in range(self.env.num_agents)]
        self.num_agents = len(self.agents)
        env_config["map_name"] = map
        self.env_config = env_config

    def reset(self):
        original_obs = self.env.reset()
        obs = {}
        for i, name in enumerate(self.agents):
            obs[name] = {"obs": np.array(original_obs[i])}
        return obs

    def step(self, action_dict):
        action_ls = [action_dict[key] for key in action_dict.keys()]
        o, r, d, info = self.env.step(action_ls)
        rewards = {}
        obs = {}
        for i, key in enumerate(action_dict.keys()):
            rewards[key] = r[i]
            obs[key] = {
                "obs": np.array(o[i])
            }
        dones = {"__all__": True if sum(d) == self.num_agents else False}
        return obs, rewards, dones, {}

    def close(self):
        self.env.close()

    def render(self, mode=None):
        self.env.render()
        time.sleep(0.05)
        return True

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.env.env_MARL.time_steps,
            "policy_mapping_info": policy_mapping_dict
        }
        return env_info


if __name__ == '__main__':
    env_config_file_path = yaml_config_file_path
    # # register new env
    ENV_REGISTRY["magym"] = RLlibMAGym

    # initialize env
    env = marl.make_env(environment_name="magym", map_name=map_name, abs_path=env_config_file_path)
    # pick algorithm
    # mappo = marl.algos.mappo(hyperparam_source="test")
    # mappo = marl.algos.ia2c(hyperparam_source="test")
    mappo = marl.algos.ippo(hyperparam_source="test")
    # customize model
    model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "256-256"})
    # start learning
    num_cores = multiprocessing.cpu_count()
    cores_per_task = 1
    training_iteration = env[1]['env_args']['iterations']
    results = mappo.fit(env, model, stop={'training_iteration': training_iteration, "timesteps_total":100000000000000},
                        local_mode=False, num_gpus=0, num_workers=cores_per_task, num_cpus_per_worker=1,
                        share_policy='all', checkpoint_freq=100)


# plot rewards in the end
    path = list(results.trial_dataframes.keys())[0]
    results_path = path + '/progress.csv'
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    import shutil
    from pathlib import Path
    df = pd.read_csv(results_path)
    curr_reward = df['episode_reward_mean'][2:].values
    ts = df['timesteps_total'][2:].values
    plt.plot(ts, curr_reward)
    plt.legend()
    plt.ylabel('Episode Reward')
    plt.xlabel('Simulation Timesteps')
    plt.title('Training curve')
    plt.savefig(path+'/training_progress_timesteps.pdf')
    plt.close()

    curr_timestamp = df['timestamp'][2:].values
    curr_relative_time = (curr_timestamp - curr_timestamp[0]) / 3600
    plt.plot(curr_relative_time, curr_reward)
    plt.legend()
    plt.ylabel('Episode Reward')
    plt.xlabel('Time [hours]')
    plt.title('MF training curve')
    plt.savefig(path+'/training_progress_time.pdf')
    plt.close()

    #get best checkpoint
    metric = "episode_reward_mean"
    mode = "max"
    best_trial = results.get_best_trial(metric=metric, mode=mode)
    value_best_metric = best_trial.metric_analysis[metric][mode]
    # Checkpoint with the lowest policy loss value:
    ckpt = results.get_best_checkpoint(
        best_trial,
        metric=metric,
        mode=mode)

    # create best result folder and copy this checkpoint there
    br = path + "/best_result/" + Path(ckpt).stem
    files = os.listdir(Path(ckpt).parent)
    shutil.copytree(Path(ckpt).parent, br)

