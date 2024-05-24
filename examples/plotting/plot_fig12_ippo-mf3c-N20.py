import fnmatch
import itertools
import os
import string
import matplotlib as mpl
from csv import reader
import csv
import sys
csv.field_size_limit(sys.maxsize)
import matplotlib.pyplot as plt
import numpy as np
import ray
from cycler import cycler
from gym.wrappers import FlattenObservation
from ray.rllib.agents.ppo import ppo
from ray.tune import register_env

from marllib.envs.m3fc_envs.experiments.args_parser import generate_config_from_kw
# from wrappers.major_minor_comms_gaussian import CommsGaussianMajorMinorMFCEnv
# from wrappers.major_minor_comms_gaussian_per_agent import CommsGaussianMajorMinorMFCEnvPerAgent
# from wrappers.major_minor_discrete_gaussian import DiscreteGaussianMajorMinorMFCEnv
# from wrappers.major_minor_discrete_gaussian_per_agent import DiscreteGaussianMajorMinorMFCEnvPerAgent
# from wrappers.major_minor_discrete_pure_tuple import PureDiscreteTupleMajorMinorMFCEnv
# from wrappers.major_minor_discrete_pure_tuple_per_agent import PureDiscreteTupleMajorMinorMFCEnvPerAgent
# from wrappers.major_minor_foraging_wrapper import ForagingMajorMinorMFCEnv
# from wrappers.major_minor_foraging_wrapper_per_agent import ForagingMajorMinorMFCEnvPerAgent
# from wrappers.comms_wrapper_gaussian import CommsObsGaussianWrapper
# from wrappers.comms_wrapper_gaussian_per_agent import CommsObsGaussianWrapperPerAgent


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
        action = trainer.compute_single_action(state)
        state, rewards, done, _ = env.step(action)
        value += np.array(rewards)
    return value, states


def evaluate_objective_N_MFC(game, iterations, solver, num_return_trials, Ns, trial):
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


    trainer = ppo.PPOTrainer(env=env_name, config=trainer_config)
    files = 'path_to_checkpoint'
    trainer.load_checkpoint(max(files, key=os.path.getctime))
    save_dir = config['exp_dir']
    J_Ns = []
    for N in [20]:
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
    np.save(save_dir + f"/J_Ns_{game, iterations, solver, num_return_trials, Ns, trial}.npy", J_Ns)

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
    skip_n = 2

    games = ['comms-two', 'major-minor-formation', 'major-minor-beach-tuple', 'major-minor-foraging', 'major-minor-potential', 'major-minor-potential']
    labels = ['2G', 'Formation', 'Beach', 'Foraging', 'Potential', 'Potential']
    solvers = ['comms', 'major-minor-comms-gaussian', 'major-minor-pure-discrete-tuple', 'major-minor-foraging', 'major-minor-gaussian', 'major-minor-gaussian']
    marl_solvers = ['marl-comms', 'major-minor-marl-comms-separate', 'major-minor-marl-pure-discrete-tuple-separate', 'major-minor-marl-foraging-separate', 'major-minor-marl-discrete-separate',]
    iterations = [500, 500, 500, 500, 500]

    iterations_CTDE = [5000, 5000, 5000, 13000, 5000]
    trials_CTDE = [0, 0, 1, 0, 0]
    solvers_CTDE = ['comms', 'major-minor-comms-gaussian', 'major-minor-pure-discrete-tuple', 'major-minor-foraging', 'major-minor-gaussian', 'major-minor-gaussian']

    ippo_marllib_path = './'

    fig = plt.figure()
    # spec = mpl.gridspec.GridSpec(ncols=6, nrows=2) # 6 columns evenly divides both 2 & 3
    for game, solver, iterations, label, iteration_CTDE, trial_CTDE, solver_CTDE in zip(games, marl_solvers, iterations, labels, iterations_CTDE, trials_CTDE, solvers_CTDE):
        if game == 'comms-two':
            exp = 'exp5'
        elif game == 'major-minor-formation':
            exp = 'exp6'
        elif    game == 'major-minor-beach-tuple':
            exp = 'exp8'
        elif    game == 'major-minor-foraging':
            exp = 'exp7'
        elif    game == 'major-minor-potential':
            exp = 'exp9'
        else:
            raise NotImplementedError
        clist = itertools.cycle(cycler(color='rbkgcmy'))
        linestyle_cycler = itertools.cycle(cycler('linestyle', ['-', '--', ':', '-.']))
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
        all_time_steps = []
        for N in [5, 10, 20]:
            all_mf_returns = []

            for id in range(3):
                file_path = ippo_marllib_path + 'ippo_mlp_{}_{}_{}_ippo'.format(exp, N, id)
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
                time_steps = []
                mf_returns = []
                files = find('progress.csv', file_path )
                with open(max(files, key=os.path.getctime), 'r') as read_obj:
                    csv_reader = reader(read_obj)

                    first_line = csv_reader.__next__()
                    for idx, key in zip(range(len(first_line)), first_line):
                        if key == 'timesteps_total':
                            idx_timesteps = idx
                        if key == 'episode_reward_mean':
                            idx_return = idx

                    for row in csv_reader:
                        time_steps.append(int(row[idx_timesteps]) // (N+1))
                        mf_returns.append(float(row[idx_return]))

                all_mf_returns.append(mf_returns)

            all_mf_returns = np.array(all_mf_returns) / (N+1)
            std_returns = np.std(all_mf_returns, axis=0)
            mean_returns = np.mean(all_mf_returns, axis=0)
            max_returns = np.amax(all_mf_returns, axis=0)

            all_time_steps.append(time_steps[-1])
            color = clist.__next__()['color']
            linestyle = linestyle_cycler.__next__()['linestyle']
            subplot.plot(time_steps[::skip_n], mean_returns[::skip_n] + std_returns[::skip_n], linestyle, color=color,
                         label='_nolabel_', alpha=0.5)
            subplot.plot(time_steps[::skip_n], mean_returns[::skip_n] - std_returns[::skip_n], linestyle, color=color,
                         label='_nolabel_', alpha=0.5)
            subplot.plot(time_steps[::skip_n], mean_returns[::skip_n], linestyle, color=color, label=f"IPPO N={N}", alpha=0.85)
            subplot.fill_between(time_steps[::skip_n], mean_returns[::skip_n] - std_returns[::skip_n],
                                 mean_returns[::skip_n] + std_returns[::skip_n], color=color, alpha=0.15)

            print(f"{game}: {N} has {np.max(max_returns)}")


        """ Plot also value of MFC at N=20 """
        num_return_trials = 100 if i < 5 else 300 if i < 6 else 500
        Ns = [2, 4, 6, 8, 10, 20, 30, 40, 50]
        J_Ns_MFC = evaluate_objective_N_MFC(game, iteration_CTDE, solver_CTDE, num_return_trials, Ns, trial_CTDE)

        std_returns = 2 * np.std(J_Ns_MFC, axis=1) / np.sqrt(num_return_trials)
        mean_returns = np.mean(J_Ns_MFC, axis=1)
        print('a', game, mean_returns, std_returns)

        color = clist.__next__()['color']
        linestyle = linestyle_cycler.__next__()['linestyle']
        N = 20

        subplot.plot([0, time_steps[-1]], [mean_returns[5]] * 2, linestyle, color=color, label=f"MF N={N}", alpha=0.85
                         ,  linewidth=5.0)

        plt.grid('on')
        plt.xlabel(r'Steps $t$', fontsize=22)
        if i==2:
            plt.ylabel(r'Return $J(\pi)$', fontsize=22)
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))


    import time
    dt = time.time_ns()
    """ Finalize plot """
    plt.gcf().set_size_inches(23, 4)
    plt.tight_layout(w_pad=-0.0, h_pad=0.0)
    lgd = plt.legend(loc="lower right", bbox_to_anchor=(1.9, -0.1))
    plt.savefig(f'ippoMM_MARL_training_curves_new{dt}.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight',
                transparent=True, pad_inches=0)


if __name__ == '__main__':
    ray.init(local_mode=True)
    plot()
