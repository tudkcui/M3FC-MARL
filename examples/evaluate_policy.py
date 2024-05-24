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
example of how to load a model and keep training for timesteps_total steps
"""

from marllib import marl

# from exp_0_cn_two_gaussians import RLlibMAGym
from marllib.envs.base_env import ENV_REGISTRY
from marllib.marl.algos.core.CC.mappo import MAPPOTrainer
from marllib.marl.algos.core.IL.ppo import IPPOTrainer
import yaml
from pathlib import Path
import pickle
import os

map_name = "exp4"
from exp_4_mm_potential import RLlibMAGym
yaml_config_file_path = os.path.join(os.path.dirname(__file__),
                                     "../marllib/envs/base_env/config/{}.yaml".format(map_name))
dir_path = './results/major-minor-potential_major-minor-gaussian_fc_5000_0_0_0_1_0_0_0_7_3_2_2_300_0/major-minor-potential_major-minor-gaussian_fc_5000_0_0_0_1_0_0_0_7_3_2_2_300_0pih2ak82/'
chkpoint_number = 3700
checkpoint_file = dir_path + "best_result/checkpoint-{}/checkpoint-{}".format(chkpoint_number, chkpoint_number)  # checkpoint path
env_config_file_path = yaml_config_file_path
algo_data = Path(dir_path + "params.pkl")

with algo_data.open('rb') as pf:
    config = pickle.load(pf)

with open(env_config_file_path, "r") as f:
    env_config_dict = yaml.load(f, Loader=yaml.FullLoader)
    f.close()

config['env_config'] = env_config_dict['env_args']

from ray import tune
tune.register_env('magym', lambda x: RLlibMAGym(x))

ENV_REGISTRY["magym"] = RLlibMAGym
env = marl.make_env(environment_name="magym", map_name=map_name, abs_path=yaml_config_file_path)

if 'MA' in map_name:
    mappo = marl.algos.mappo(hyperparam_source="test")
else:
    mappo = marl.algos.ippo(hyperparam_source="test")

nn_size = config['model']['custom_model_config']['model_arch_args']['encode_layer']
# model = marl.build_model(env, mappo, {"core_arch": "lstm", "encode_layer": "256-64"})
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": nn_size})

from ray.rllib.models import ModelCatalog
ModelCatalog.register_custom_model("Centralized_Critic_Model", model[0])
ModelCatalog.register_custom_model("Base_Model", model[0])

config['num_workers'] = 1
config['num_gpus'] = 0
# agent = MAPPOTrainer(config=config, env="magym")
agent = IPPOTrainer(config=config, env="magym")
agent.restore(checkpoint_file)
test_env = env[0]

obs = test_env.reset()  # create a test env
action_all = {}

for j in range(50):
    for i in range(test_env.num_agents):
        o = obs['agent_{}'.format(i)]
        # action = agent.compute_single_action(o)
        action, _ = agent.compute_single_action(o)    # save state of each agent separately for next time
        action_all['agent_{}'.format(i)] = action
    obs, reward, done, info = test_env.step(action_all)


