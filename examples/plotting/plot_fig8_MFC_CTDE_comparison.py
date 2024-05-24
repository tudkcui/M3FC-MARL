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

from experiments.args_parser import generate_config_from_kw
from wrappers.major_minor_comms_gaussian import CommsGaussianMajorMinorMFCEnv
from wrappers.major_minor_comms_gaussian_per_agent import CommsGaussianMajorMinorMFCEnvPerAgent
from wrappers.major_minor_discrete_gaussian import DiscreteGaussianMajorMinorMFCEnv
from wrappers.major_minor_discrete_gaussian_per_agent import DiscreteGaussianMajorMinorMFCEnvPerAgent
from wrappers.major_minor_discrete_pure_tuple import PureDiscreteTupleMajorMinorMFCEnv
from wrappers.major_minor_discrete_pure_tuple_per_agent import PureDiscreteTupleMajorMinorMFCEnvPerAgent
from wrappers.major_minor_foraging_wrapper import ForagingMajorMinorMFCEnv
from wrappers.major_minor_foraging_wrapper_per_agent import ForagingMajorMinorMFCEnvPerAgent
from wrappers.comms_wrapper_gaussian import CommsObsGaussianWrapper
from wrappers.comms_wrapper_gaussian_per_agent import CommsObsGaussianWrapperPerAgent


def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


def run_once_MFC(env, trainer):
    done = 0
    obs_space = env.observation_space
    state = env.reset()
    value = 0
    states = []
    while not done:
        states.append(state)
        action = trainer.compute_action(state)
        state, rewards, done, _ = env.step(action)
        value += np.array(rewards)
    return value, states


def run_once_open_loop(env, actions):
    done = 0
    obs_space = env.observation_space
    state = env.reset()
    value = 0
    states = []
    for action in actions:
        states.append(state)
        state, rewards, done, _ = env.step(action)
        value += np.array(rewards)
    return value, states


def evaluate_objective_N_MFC(game, iterations, solver, num_return_trials, Ns, trial):

    if game == 'comms-two':
        map_name = 'exp0'
    elif game == 'major-minor-formation':
        map_name = 'exp1'
    elif game == 'major-minor-beach-tuple':
        map_name = 'exp3'
    elif game == 'major-minor-foraging':
        map_name = 'exp2'
    elif game ==  'major-minor-potential':
        map_name = 'exp4'
    else:
        raise NotImplementedError

    config = generate_config_from_kw(**{
        'game': game,
        'solver': solver,
        'cores': 1,
        'policy': 'fc',
        'iterations': iterations,
        'id': trial,
        'verbose': 0,
        'use_lstm': 0,
        'dims': 1 if game == 'major-minor-potential' else 2,
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

    def env_creator(env_config=None):
        if config['policy'] is None:
            # Fully connected wrapper
            return FlattenObservation(config['solver'](config['game'](**config), **config))
        else:
            return config['solver'](config['game'](**config), **config)

    env_name = config['game'].__name__ + hash(frozenset(config.items())).__str__() + "tmp"
    register_env(env_name, env_creator)
    if config['policy'] is not None:
        trainer_config['model'] = {
            "custom_model": config['policy'],
            "use_lstm": config['use_lstm'],
            "lstm_cell_size": 64,
        }
    elif config['use_lstm']:
        trainer_config['model'] = {
            "use_lstm": True,
            "lstm_cell_size": 64,
        }

    from examples.plotting.create_NJ_files_ippo import load_trainer
    from pathlib import Path
    chkpoint_number = 5000
    f = 'checkpoint_file_path'
    dirpath = str(Path(f).parent.parent)
    trainer = load_trainer(dirpath, chkpoint_number, map_name)

    J_Ns = []
    for N in Ns:
        config = generate_config_from_kw(**{
            'game': game,
            'solver': solver,
            'cores': 1,
            'policy': 'fc',
            'iterations': iterations,
            'id': trial,
            'verbose': 0,
            'use_lstm': 0,
            'dims': 1 if game == 'major-minor-potential' else 2,
            'unbounded_actions': 0,
            'hyperparam_config': 0,
            'obs_time': 0,
            'num_agents': N,

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

        def env_creator(env_config=None):
            if config['policy'] is None:
                # Fully connected wrapper
                return FlattenObservation(config['solver'](config['game'](**config), **config))
            else:
                return config['solver'](config['game'](**config), **config)

        env_name = config['game'].__name__ + hash(frozenset(config.items())).__str__()
        register_env(env_name, env_creator)

        returns = []
        env = env_creator()

        for _ in range(num_return_trials):
            value, states = run_once_MFC(env, trainer)
            returns.append(value)
            print(f'{N}: {value}')
        J_Ns.append(returns)
    J_Ns = np.array(J_Ns)
    save_dir = ''
    np.save( save_dir + f"/J_Ns_cen_{game, iterations, solver, num_return_trials, Ns, trial}.npy",
        J_Ns)

    return J_Ns


def evaluate_objective_N_MFC_decentralized(game, iterations, solver, num_return_trials, Ns, trial, map_name):
    if game == 'comms-two':
        map_name = 'exp0'
    elif game == 'major-minor-formation':
        map_name = 'exp1'
    elif game == 'major-minor-beach-tuple':
        map_name = 'exp3'
    elif game == 'major-minor-foraging':
        map_name = 'exp2'
    elif game ==  'major-minor-potential':
        map_name = 'exp4'
    else:
        raise NotImplementedError

    config = generate_config_from_kw(**{
        'game': game,
        'solver': solver,
        'cores': 1,
        'policy': 'fc',
        'iterations': iterations,
        'id': trial,
        'verbose': 0,
        'use_lstm': 0,
        'dims': 1 if game == 'major-minor-potential' else 2,
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

    def env_creator_MFC(env_config=None):
        if config['policy'] is None:
            # Fully connected wrapper
            return FlattenObservation(config['solver'](config['game'](**config), **config))
        else:
            return config['solver'](config['game'](**config), **config)

    env_name = config['game'].__name__ + hash(frozenset(config.items())).__str__() + "tmp"
    register_env(env_name, env_creator_MFC)
    if config['policy'] is not None:
        trainer_config['model'] = {
            "custom_model": config['policy'],
            "use_lstm": config['use_lstm'],
            "lstm_cell_size": 64,
        }
    elif config['use_lstm']:
        trainer_config['model'] = {
            "use_lstm": True,
            "lstm_cell_size": 64,
        }

    from examples.plotting.create_NJ_files_ippo import load_trainer
    from pathlib import Path
    chkpoint_number = 5000
    f = 'checkpoint_file_path'
    dirpath = str(Path(f).parent.parent)
    trainer = load_trainer(dirpath, chkpoint_number, map_name)

    J_Ns = []
    for N in Ns:
        config = generate_config_from_kw(**{
            'game': game,
            'solver': solver,
            'cores': 1,
            'policy': 'fc',
            'iterations': iterations,
            'id': trial,
            'verbose': 0,
            'use_lstm': 0,
            'dims': 1 if game == 'major-minor-potential' else 2,
            'unbounded_actions': 0,
            'hyperparam_config': 0,
            'obs_time': 0,
            'num_agents': N,

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

        config['solver'] = CommsObsGaussianWrapperPerAgent if config['solver'] == CommsObsGaussianWrapper \
            else CommsGaussianMajorMinorMFCEnvPerAgent if config['solver'] == CommsGaussianMajorMinorMFCEnv \
            else PureDiscreteTupleMajorMinorMFCEnvPerAgent if config['solver'] == PureDiscreteTupleMajorMinorMFCEnv \
            else ForagingMajorMinorMFCEnvPerAgent if config['solver'] == ForagingMajorMinorMFCEnv \
            else DiscreteGaussianMajorMinorMFCEnvPerAgent if config['solver'] == DiscreteGaussianMajorMinorMFCEnv \
            else None

        def env_creator(env_config=None):
            if config['policy'] is None:
                # Fully connected wrapper
                return FlattenObservation(config['solver'](config['game'](**config), **config))
            else:
                return config['solver'](config['game'](**config), **config)

        env_name = config['game'].__name__ + hash(frozenset(config.items())).__str__()
        register_env(env_name, env_creator)

        returns = []
        env = env_creator()

        for _ in range(num_return_trials):
            env = env_creator_MFC()
            done = 0
            state = env.reset()
            value = 0
            states = []
            actions = []
            while not done:
                states.append(state)
                action = [trainer.compute_action(state) for _ in range(env.env_MARL.num_agents)]
                actions.append(action)
                state, rewards, done, _ = env.step(action)
                value += np.array(rewards)

            returns.append(value)
            print(f'{N}: {value}')
        J_Ns.append(returns)
    J_Ns = np.array(J_Ns)
    save_dir = ''
    np.save(save_dir + f"/J_Ns_dec_{game, iterations, solver, num_return_trials, Ns, trial}.npy",
        J_Ns)
    return J_Ns


def plot():
    """ Plot figures """
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.size": 24,
        "font.sans-serif": ["Helvetica"],
    })

    i = 1
    skip_n = 20


    games = ['comms-two', 'major-minor-formation', 'major-minor-beach-tuple', 'major-minor-foraging', 'major-minor-potential', 'major-minor-potential']
    labels = ['2G', 'Formation', 'Beach', 'Foraging', 'Potential', 'Potential']
    solvers = ['comms', 'major-minor-comms-gaussian', 'major-minor-pure-discrete-tuple', 'major-minor-foraging', 'major-minor-gaussian', 'major-minor-gaussian']
    iterations = [5000, 5000, 5000, 5000, 5000]
    trials = [0, 0, 1, 0, 0]

    fig = plt.figure()
    spec = mpl.gridspec.GridSpec(ncols=6, nrows=2) # 6 columns evenly divides both 2 & 3

    for game, solver, iterations, label, trial in zip(games, solvers, iterations, labels, trials):
        print(game, solver, iterations, label, trial)
        clist = itertools.cycle(cycler(color='rbgcmyk'))
        linestyle_cycler = itertools.cycle(cycler('linestyle', ['-', '--', ':', '-.']))
        # subplot = plt.subplot(2, 3, i)
        subplot = fig.add_subplot(spec[0, 0:2]) if i == 1 else \
            fig.add_subplot(spec[0, 2:4]) if i == 2 else \
            fig.add_subplot(spec[0, 4:]) if i == 3 else \
            fig.add_subplot(spec[1, 1:3]) if i == 4 else \
            fig.add_subplot(spec[1, 3:5])

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

        num_return_trials = 100 if i < 5 else 300 if i < 6 else 500
        Ns = [2, 4, 6, 8, 10, 20, 30, 40, 50]

        Ns_MF = [500]
        J_Ns_MFC = evaluate_objective_N_MFC(game, iterations, solver, num_return_trials, Ns_MF, trial)

        std_returns = 2 * np.std(J_Ns_MFC, axis=1)[0] / np.sqrt(num_return_trials)
        mean_returns = np.mean(J_Ns_MFC, axis=1)[0]

        color = clist.__next__()['color']
        linestyle = linestyle_cycler.__next__()['linestyle']
        subplot.plot(Ns, [mean_returns + std_returns] * len(Ns), linestyle, color=color, label='_nolabel_', alpha=0.5)
        subplot.plot(Ns, [mean_returns - std_returns] * len(Ns), linestyle, color=color, label='_nolabel_', alpha=0.5)
        subplot.plot(Ns, [mean_returns] * len(Ns), linestyle, color=color, label=r'MF', alpha=0.85)
        subplot.fill_between(Ns, [mean_returns - std_returns] * len(Ns), [mean_returns + std_returns] * len(Ns),
                             color=color, alpha=0.15)

        J_Ns_MFC = evaluate_objective_N_MFC(game, iterations, solver, num_return_trials, Ns, trial)

        std_returns = 2 * np.std(J_Ns_MFC, axis=1) / np.sqrt(num_return_trials)
        mean_returns = np.mean(J_Ns_MFC, axis=1)

        color = clist.__next__()['color']
        linestyle = linestyle_cycler.__next__()['linestyle']
        subplot.plot(Ns, mean_returns + std_returns, linestyle, color=color, label='_nolabel_', alpha=0.5)
        subplot.plot(Ns, mean_returns - std_returns, linestyle, color=color, label='_nolabel_', alpha=0.5)
        subplot.plot(Ns, mean_returns, linestyle, color=color, label="CE", alpha=0.85)
        subplot.fill_between(Ns, mean_returns - std_returns, mean_returns + std_returns, color=color, alpha=0.15)

        # """ DE """
        J_Ns_MFCs_open_loop = evaluate_objective_N_MFC_decentralized(game, iterations, solver, num_return_trials, Ns, trial)

        std_returns_MFC_open_loop = 2 * np.std(J_Ns_MFCs_open_loop, axis=1) / np.sqrt(num_return_trials)
        mean_returns_MFC_open_loop = np.mean(J_Ns_MFCs_open_loop, axis=1)

        color = clist.__next__()['color']
        linestyle = linestyle_cycler.__next__()['linestyle']
        subplot.plot(Ns, mean_returns_MFC_open_loop + std_returns_MFC_open_loop, linestyle, color=color, label='_nolabel_', alpha=0.5)
        subplot.plot(Ns, mean_returns_MFC_open_loop - std_returns_MFC_open_loop, linestyle, color=color, label='_nolabel_', alpha=0.5)
        subplot.plot(Ns, mean_returns_MFC_open_loop, linestyle, color=color, label="DE", alpha=0.85)
        subplot.fill_between(Ns, mean_returns_MFC_open_loop - std_returns_MFC_open_loop, mean_returns_MFC_open_loop + std_returns_MFC_open_loop,
                             color=color, alpha=0.15)

        plt.grid('on')
        plt.xlabel(r'Num agents $N$', fontsize=22)
        if i==2 or i==5:
            plt.ylabel(r'Return $J_N(\pi)$', fontsize=22)
        plt.xlim([min(Ns),  max(Ns)])

    """ Finalize plot """
    plt.gcf().set_size_inches(13, 6)
    plt.tight_layout(w_pad=-0.15, h_pad=0.05)
    plt.legend(loc="lower right", bbox_to_anchor=(1.75, -0.1))
    plt.savefig(f'MM_CTDE_comparisons_all.pdf', bbox_inches='tight', transparent=True, pad_inches=0)
    plt.show()


if __name__ == '__main__':
    ray.init(local_mode=True)
    plot()
