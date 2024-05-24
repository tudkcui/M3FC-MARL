import fnmatch
import itertools
import os
import string
import matplotlib as mpl
from csv import reader

import matplotlib.pyplot as plt
import numpy as np
import ray
from cycler import cycler
from gym.wrappers import FlattenObservation
from ray.rllib.agents.ppo import ppo
from ray.tune import register_env

from marllib.envs.m3fc_envs.args_parser import generate_config_from_kw
# from marllib.envs.m3fc_envs.wrappers.major_minor_comms_gaussian import CommsGaussianMajorMinorMFCEnv
# from marllib.envs.m3fc_envs.wrappers.major_minor_comms_gaussian_per_agent import CommsGaussianMajorMinorMFCEnvPerAgent
# from marllib.envs.m3fc_envs.wrappers.major_minor_discrete_gaussian import DiscreteGaussianMajorMinorMFCEnv
# from marllib.envs.m3fc_envs.wrappers.major_minor_discrete_gaussian_per_agent import DiscreteGaussianMajorMinorMFCEnvPerAgent
# from marllib.envs.m3fc_envs.wrappers.major_minor_discrete_pure_tuple import PureDiscreteTupleMajorMinorMFCEnv
# from marllib.envs.m3fc_envs.wrappers.major_minor_discrete_pure_tuple_per_agent import PureDiscreteTupleMajorMinorMFCEnvPerAgent
# from marllib.envs.m3fc_envs.wrappers.major_minor_foraging_wrapper import ForagingMajorMinorMFCEnv
# from marllib.envs.m3fc_envs.wrappers.major_minor_foraging_wrapper_per_agent import ForagingMajorMinorMFCEnvPerAgent
from marllib.envs.m3fc_envs.wrappers.comms_wrapper_gaussian import CommsObsGaussianWrapper
from marllib.envs.m3fc_envs.wrappers.comms_wrapper_gaussian_per_agent import CommsObsGaussianWrapperPerAgent

from gym import spaces

import sys
import csv

csv.field_size_limit(sys.maxsize)
def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result



def plot():

    """ Plot figures """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 24,
        "font.sans-serif": ["Helvetica"],
    })

    i = 1
    skip_n = 2

    games = ['comms-two', 'major-minor-formation', 'major-minor-beach-tuple', 'major-minor-foraging', 'major-minor-potential', 'major-minor-potential']
    labels = ['2G', 'Formation', 'Beach', 'Foraging', 'Potential', 'Potential']
    solvers = ['comms', 'major-minor-comms-gaussian', 'major-minor-pure-discrete-tuple', 'major-minor-foraging', 'major-minor-gaussian', 'major-minor-gaussian']
    marl_solvers = ['marl-comms', 'major-minor-marl-comms-separate', 'major-minor-marl-pure-discrete-tuple-separate', 'major-minor-marl-foraging-separate', 'major-minor-marl-discrete-separate',]
    iterations = [500, 500, 500, 500, 500]
    solvers_CTDE = ['comms', 'major-minor-comms-gaussian', 'major-minor-pure-discrete-tuple', 'major-minor-foraging', 'major-minor-gaussian', 'major-minor-gaussian']
    trials_CTDE = [ 0,0, 1, 0, 0]
    iterations_CTDE = [5000,5000,5000,5000,5000]


    games = ['major-minor-formation', 'major-minor-beach-tuple', 'major-minor-foraging', 'major-minor-potential', 'major-minor-potential']
    labels = ['Formation', 'Beach', 'Foraging', 'Potential', 'Potential']
    solvers = [ 'major-minor-comms-gaussian', 'major-minor-pure-discrete-tuple', 'major-minor-foraging', 'major-minor-gaussian', 'major-minor-gaussian']
    marl_solvers = [ 'major-minor-marl-comms-separate', 'major-minor-marl-pure-discrete-tuple-separate', 'major-minor-marl-foraging-separate', 'major-minor-marl-discrete-separate',]
    iterations = [ 500, 500, 500, 500]
    solvers_CTDE = [ 'major-minor-comms-gaussian', 'major-minor-pure-discrete-tuple', 'major-minor-foraging', 'major-minor-gaussian', 'major-minor-gaussian']
    trials_CTDE = [ 0, 1, 0, 0]
    iterations_CTDE = [5000,5000,5000,5000]

    exp_list = ['exp5', 'exp6',  'exp8', 'exp7','exp9']
    main_path = './exp_results/results/'


    all_mf_return = []
    for exp in exp_list:
        exp_return = []
        for id in range(3):
            print(exp)
            file_name = 'mappo_mlp_{}_20_{}_mappo'.format(exp, id)
            file_path = main_path + file_name
            print(file_path)
            files = find('progress.csv', file_path)
            print(files)
            with open(max(files, key=os.path.getctime), 'r') as read_obj:
                csv_reader = reader(read_obj)

                first_line = csv_reader.__next__()
                for idx, key in zip(range(len(first_line)), first_line):
                    if key == 'timesteps_total':
                        idx_timesteps = idx
                    if key == 'episode_reward_mean':
                        idx_return = idx

                # for row in csv_reader:
                #     time_steps.append(int(row[idx_timesteps]) // (N+1))
                #     mf_returns.append(float(row[idx_return]))
                for row in csv_reader:
                    time_steps = (int(row[idx_timesteps]))
                    mf_returns = (float(row[idx_return]))

            mf_returns = np.array(mf_returns)/21
            exp_return.append(mf_returns)

        all_mf_return.append(exp_return)
        print(all_mf_return)
        print(all_mf_return)

if __name__ == '__main__':
    # ray.init(local_mode=True)
    plot()
