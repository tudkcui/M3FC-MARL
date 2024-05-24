import fnmatch
import itertools
import os
import string
from csv import reader
import matplotlib as mpl

import matplotlib.pyplot as plt
import numpy as np
import ray
from cycler import cycler

from marllib.envs.m3fc_envs.args_parser import generate_config_from_kw


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
    skip_n = 8

    games = ['comms-two', 'major-minor-formation', 'major-minor-beach-tuple', 'major-minor-foraging', 'major-minor-potential', 'major-minor-potential']
    labels = ['2G', 'Formation', 'Beach', 'Foraging', 'Potential', 'Potential']
    solvers = ['comms', 'major-minor-comms-gaussian', 'major-minor-pure-discrete-tuple', 'major-minor-foraging', 'major-minor-gaussian', 'major-minor-gaussian']
    iterations = [5000, 5000, 5000, 13000, 5000]

    fig = plt.figure()
    spec = mpl.gridspec.GridSpec(ncols=6, nrows=2) # 6 columns evenly divides both 2 & 3

    for game, solver, iterations, label in zip(games, solvers, iterations, labels):
        print(game, solver, iterations, label)
        clist = itertools.cycle(cycler(color='rbkgcmy'))
        linestyle_cycler = itertools.cycle(cycler('linestyle', ['-', '--', ':', '-.']))
        # subplot = plt.subplot(2, 3, i) if i < 4 else plt.subplot(2, 2, i-1)
        subplot = plt.subplot(1, 5, i)
        # subplot = fig.add_subplot(spec[0, 0:2]) if i==1 else \
        #     fig.add_subplot(spec[0, 2:4]) if i==2 else \
        #     fig.add_subplot(spec[0, 4:]) if i==3 else \
        #     fig.add_subplot(spec[1, 1:3]) if i==4 else \
        #     fig.add_subplot(spec[1, 3:5])

        # subplot.text(-0.01, 1.06, '(' + string.ascii_lowercase[i - 1] + ')', transform=subplot.transAxes, weight='bold')
        subplot.annotate('(' + string.ascii_lowercase[i - 1] + ')',
        (1, 0),
        xytext=(-36, +32),
        xycoords='axes fraction',
        textcoords='offset points',
        fontweight='bold',
        color='black',
        backgroundcolor='white',
        ha='left', va='top')
        i += 1

        all_mf_returns = []

        for id in range(3):
            config = generate_config_from_kw(**{
                'game': game,
                'solver': solver,
                'cores': 1,
                'policy': 'fc',
                'iterations': iterations,
                'id': id,
                'verbose': 0,
                'use_lstm': 0,
                'dims': 1 if i == 6 else 2,
                'unbounded_actions': 0,
                'hyperparam_config': 0,
                'obs_time': 0,
                'num_agents': 300,

                'x_bins_per_dim': 7,
                'u_bins_per_dim': 3,
                'num_GP_points': 2,
                'num_KDE_points': 2,
                'problem_config': 0,
                'scale_rewards': 0,
            })

            trainer_config = {
                'num_workers': 1,
                "num_cpus_per_worker": 1,
                "gamma": 0.99,
                "clip_param": 0.2,
                "kl_target": 0.03,
                "normalize_actions": not config['unbounded_actions'],
                "no_done_at_end": True,
                "framework": 'torch',
            }
            if config['hyperparam_config'] == 1:
                trainer_config['train_batch_size'] = 4000
                trainer_config['sgd_minibatch_size'] = 1000
                trainer_config['num_sgd_iter'] = 5

            time_steps = []
            mf_returns = []
            files = find('progress.csv', './' + config['exp_dir'])
            with open(max(files, key=os.path.getctime), 'r') as read_obj:
                csv_reader = reader(read_obj)

                first_line = csv_reader.__next__()
                for idx, key in zip(range(len(first_line)), first_line):
                    if key == 'timesteps_total':
                        idx_timesteps = idx
                    if key == 'episode_reward_mean':
                        idx_return = idx

                for row in csv_reader:
                    time_steps.append(int(row[idx_timesteps]))
                    mf_returns.append(float(row[idx_return]))

            all_mf_returns.append(mf_returns)

        max_len = min([len(returns) for returns in all_mf_returns])
        all_mf_returns = np.array([returns[:max_len] for returns in all_mf_returns])
        std_returns = np.std(all_mf_returns, axis=0)
        mean_returns = np.mean(all_mf_returns, axis=0)
        max_returns = np.amax(all_mf_returns, axis=0)

        print(f"{game} best: {np.argmax(all_mf_returns[:, max_len-1])}")
        color = clist.__next__()['color']
        linestyle = linestyle_cycler.__next__()['linestyle']
        subplot.plot(time_steps[:max_len-1:skip_n], mean_returns[:max_len-1:skip_n] + std_returns[:max_len-1:skip_n], linestyle, color=color,
                     label='_nolabel_', alpha=0.5)
        subplot.plot(time_steps[:max_len-1:skip_n], mean_returns[:max_len-1:skip_n] - std_returns[:max_len-1:skip_n], linestyle, color=color,
                     label='_nolabel_', alpha=0.5)
        subplot.plot(time_steps[:max_len-1:skip_n], mean_returns[:max_len-1:skip_n], linestyle, color=color, label='mean', alpha=0.85)
        subplot.fill_between(time_steps[:max_len-1:skip_n], mean_returns[:max_len-1:skip_n] - std_returns[:max_len-1:skip_n],
                             mean_returns[:max_len-1:skip_n] + std_returns[:max_len-1:skip_n], color=color, alpha=0.15)
        color = clist.__next__()['color']
        subplot.plot(time_steps[:max_len-1:skip_n], max_returns[:max_len-1:skip_n], linestyle, color=color, label='max', alpha=0.85)

        print(f"{game}: has max mean {np.max(mean_returns)} ")

        plt.grid('on')
        plt.xlabel(r'Steps $t$', fontsize=22)
        if i == 2:
            plt.ylabel(r'Return $J(\pi)$', fontsize=22)
        plt.xlim([0, time_steps[max_len-1]])
    import time
    dt = time.time_ns()
    """ Finalize plot """
    plt.gcf().set_size_inches(21, 3)
    plt.tight_layout(w_pad=-0.0, h_pad=0.0)
    plt.legend(loc="lower right", bbox_to_anchor=(1.75, -0.1))
    plt.savefig(f'M3FC_training_curves_{dt}.pdf', bbox_inches='tight', transparent=True, pad_inches=0)


if __name__ == '__main__':
    ray.init(local_mode=True)
    # ray.init(address="auto", local_mode=True)
    plot()