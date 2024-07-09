# Major-Minor Mean Field Multi-Agent Reinforcement Learning
This is the official repository for multi-agent reinforcement learning (MARL) based on major-minor mean field control. It formally reduces certain MARL problems to single-agent reinforcement learning (RL), and then applies single-agent policy gradient methods (such as PPO) as a MARL algorithm. The implementation is based on [MARLlib](https://github.com/Replicable-MARL/MARLlib) and writing a wrapper for MFC-type MARL problems, turning them into single-agent RL problems.

## Setup

Create a python==3.8.17 environment, e.g.
```shell script
python3.8 -m venv venv
source venv/bin/activate
```
 
Install requirements:

```shell script
pip install -r requirements_m3fc.txt
```

Then downgrade first two packages and install last two packages for MARLlib:
```shell script
pip install wheel==0.38.4
pip install setuptools==66.0.0
pip install gym==0.20.0
pip install ma-gym==0.0.14
```

For MARLlib's integrated RLlib patch, install older protobuf version
```shell script
pip install protobuf==3.20.2
```

Finally, patch RLlib using MARLlib's patch
```shell script
python marllib/patch/add_patch.py -y
```

## Training

To train, run the scripts in the examples folder, e.g. using Pycharm, or alternatively

```shell script
cd ./examples
python ./exp_0_cn_two_gaussians.py
```

Results are saved in ./examples/exp_results

In the scripts, comment in the algorithm you want to use.

The environment config yaml files are in the marllib/envs/base_env/config folder.

For example to change the parameters of the env used by exp_0_cn_two_gaussians, edit the exp0.yaml file.

The training config yaml file is in the folder marllib/marl/algos/hyperparams/test. We have used mappo.yaml. ippo.yaml and ia2c.yaml.

Their corresponding scripts are in the folder marllib/marl/algos/scripts.

Once trained policies are available, the plotting can be done using the scripts in .examples/plotting folder.
