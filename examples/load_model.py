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
# from examples.exp_0_cn_two_gaussians import RLlibMAGym
import os
from marllib.envs.base_env import ENV_REGISTRY

# map_name = "Comm2G"
# yaml_config_file_path = os.path.join(os.path.dirname(__file__),
#                                      "../marllib/envs/base_env/config/{}.yaml".format(map_name))

map_name = "exp6_20_0_mappo"
from examples.all_python_files.exp6_20_0_mappo import RLlibMAGym
yaml_config_file_path = os.path.join(os.path.dirname(__file__),
                                     "../marllib/envs/base_env/config/all_yaml_files/{}.yaml".format(map_name))
                                     # "../../marllib/envs/base_env/config/all_yaml_files/{}.yaml".format(map_name))
env_config_file_path = yaml_config_file_path
# # register new env
from ray import tune
tune.register_env('magym', lambda x: RLlibMAGym(x))

ENV_REGISTRY["magym"] = RLlibMAGym

# REGISTRY["Comm2G"] = gymEnv
# initialize env
env = marl.make_env(environment_name="magym", map_name=map_name, abs_path=env_config_file_path)

# pick ppo algorithms
if 'mappo' in map_name:
    mappo = marl.algos.mappo(hyperparam_source="test")
else:
    mappo = marl.algos.ippo(hyperparam_source="test")
# customize model
model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "256-256"})

# # initialize algorithm and load hyperparameters
# mappo = marl.algos.mappo(hyperparam_source="test")
# # build agent model based on env + algorithms + user preference if checked available
# model = marl.build_model(env, mappo, {"core_arch": "mlp", "encode_layer": "128-128"})

# rendering
mappo.render(env, model,
             # stop={'timesteps_total': 100000000000000},
             stop={'training_iteration': 500, 'timesteps_total': 100000000000000},
             restore_path={'params_path':
                            "exp_results/checkpoints_to_restore/20230920_with_checkpoints/mappo_mlp_exp6_20_0_mappo/mappo_mlp_exp6_20_0_mappo/MAPPOTrainer_magym_exp6_20_0_mappo_11ef5_00000_0_2023-09-11_10-56-17/params.json",
                           'model_path':
                           'exp_results/checkpoints_to_restore/20230920_with_checkpoints/mappo_mlp_exp6_20_0_mappo/mappo_mlp_exp6_20_0_mappo/MAPPOTrainer_magym_exp6_20_0_mappo_11ef5_00000_0_2023-09-11_10-56-17/checkpoint_000332/checkpoint-332'
                           },
             num_workers=59,
             local_mode=False,
             share_policy="all",
             checkpoint_end=True,
             num_gpus=0,
             checkpoint_freq=1
             )

print()
