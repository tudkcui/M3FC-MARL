import argparse

from marllib.envs.m3fc_envs.comms_two_gaussians import CommsTwoGaussiansEnv
from marllib.envs.m3fc_envs.major_minor_beach_tuple import MajorMinorBeachTupleEnv
from marllib.envs.m3fc_envs.major_minor_foraging import MajorMinorForagingEnv
from marllib.envs.m3fc_envs.major_minor_formation import MajorMinorFormationEnv
from marllib.envs.m3fc_envs.major_minor_potential_field import MajorMinorPotentialFieldEnv
from marllib.envs.m3fc_envs.wrappers.comms_wrapper_gaussian import CommsObsGaussianWrapper
from marllib.envs.m3fc_envs.wrappers.major_minor_comms_gaussian import CommsGaussianMajorMinorMFCEnv
from marllib.envs.m3fc_envs.wrappers.major_minor_discrete_gaussian import DiscreteGaussianMajorMinorMFCEnv
from marllib.envs.m3fc_envs.wrappers.major_minor_discrete_pure_tuple import PureDiscreteTupleMajorMinorMFCEnv
from marllib.envs.m3fc_envs.wrappers.major_minor_foraging_wrapper import ForagingMajorMinorMFCEnv
from marllib.envs.m3fc_envs.wrappers.major_minor_marl_obs_wrapper_comms_separate import MajorMinorMARLObsWrapperCommsSeparate
from marllib.envs.m3fc_envs.wrappers.major_minor_marl_obs_wrapper_discrete_pure_tuple_separate import \
    MajorMinorMARLObsWrapperDiscretePureTupleSeparate
from marllib.envs.m3fc_envs.wrappers.major_minor_marl_obs_wrapper_discrete_separate import MajorMinorMARLObsWrapperDiscreteSeparate
from marllib.envs.m3fc_envs.wrappers.major_minor_marl_obs_wrapper_foraging_separate import MajorMinorMARLObsWrapperForagingSeparate
from marllib.envs.m3fc_envs.wrappers.marl_obs_wrapper_comms import MARLObsWrapperComms


def parse_args():
    parser = argparse.ArgumentParser(description="MFC")
    parser.add_argument('--cores', type=int, help='number of cores per worker', default='1')
    parser.add_argument('--game', help='game to solve')
    parser.add_argument('--solver', help='solver used')
    parser.add_argument('--policy', help='policy used', default='fc')
    parser.add_argument('--iterations', type=int, help='number of training iterations', default=10000)
    parser.add_argument('--id', type=int, help='experiment id', default=0)
    parser.add_argument('--verbose', type=int, help='debug outputs', default=0)
    parser.add_argument('--use_lstm', type=int, help='use lstm policy', default=0)
    parser.add_argument('--dims', type=int, help='dimensionality of problem', default=2)
    parser.add_argument('--unbounded_actions', type=int, help='unbounded action spaces', default=0)
    parser.add_argument('--hyperparam_config', type=int, help='setting the PPO parameters', default=0)
    parser.add_argument('--num_agents', type=int, help='Number of agents simulated', default=300)
    parser.add_argument('--x_bins_per_dim', type=int, help='number of state discretization bins', default=3)
    parser.add_argument('--u_bins_per_dim', type=int, help='number of action discretization bins', default=3)

    parsed, unknown = parser.parse_known_args()
    def isfloat(num):
        try:
            float(num)
            return True
        except ValueError:
            return False
    def isint(num):
        try:
            int(num)
            return True
        except ValueError:
            return False
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg.split('=')[0],
                                type=int if isint(arg.split('=')[1]) else float if isfloat(arg.split('=')[1]) else str)

    return parser.parse_args()


def generate_config(args):
    new_args = {}
    # new_args['cores'] = args['env_args'].get('cores', 1)    #'number of cores per worker'
    new_args['game'] = args['env_args'].get('game')    #'game to solve'
    new_args['solver'] = args['env_args'].get('solver')    #solver used
    new_args['policy'] = args['env_args'].get('policy', 'fc')    #solver used
    new_args['solver'] = args['env_args'].get('solver')    #policy used
    new_args['iterations'] = args['env_args'].get('iterations')    #number of training iterations
    new_args['id'] = args['env_args'].get('id', 0)    #experiment id
    # new_args['seed'] = args['env_args'].get('seed', 0)    #experiment id
    new_args['verbose'] = args['env_args'].get('verbose', 0)    #ndebug outputs
    new_args['use_lstm'] = args['env_args'].get('use_lstm', 0)    #use lstm policy'
    new_args['dims'] = args['env_args'].get('dims', 2)    #dimensionality of problem
    new_args['use_lstm'] = args['env_args'].get('use_lstm', 0)    #use lstm policy'
    new_args['unbounded_actions'] = args['env_args'].get('unbounded_actions', 0)    #unbounded action spaces
    new_args['hyperparam_config'] = args['env_args'].get('hyperparam_config', 0)    #setting the PPO parameters
    new_args['num_agents'] = args['env_args'].get('num_agents', 0)    #Number of agents simulated
    new_args['x_bins_per_dim'] = args['env_args'].get('x_bins_per_dim', 3)    #number of state discretization bins
    new_args['u_bins_per_dim'] = args['env_args'].get('u_bins_per_dim', 3)    #number of action discretization bins
    new_args['obs_time'] = args['env_args'].get('obs_time', 0)    #number of action discretization bins
    # new_args['problem_config'] = args['env_args'].get('problem_config', 0)    #
    # new_args['scale_rewards'] = args['env_args'].get('scale_rewards', 0)    #
    # new_args['num_GP_points'] = args['env_args'].get('num_GP_points', 2)    #
    # new_args['num_KDE_points'] = args['env_args'].get('num_KDE_points', 2)    #



    return generate_config_from_kw(**new_args)


def generate_config_from_kw(**kwargs):
    kwargs['exp_dir'] = "%s_%s_%s_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d" \
               % (kwargs['game'], kwargs['solver'], kwargs['policy'], kwargs['iterations'],
                  kwargs['use_lstm'], kwargs['obs_time'], kwargs['unbounded_actions'], kwargs['dims'],
                  kwargs['hyperparam_config'], kwargs['x_bins_per_dim'], kwargs['u_bins_per_dim'],
                  kwargs['num_agents'], kwargs['id'],
                  )

    # kwargs['exp_dir'] = "%s_%s_%s_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d" \
    #                     % (kwargs['game'], kwargs['solver'], kwargs['policy'], kwargs['iterations'],
    #                        kwargs['use_lstm'], kwargs['obs_time'], kwargs['unbounded_actions'], kwargs['dims'],
    #                        kwargs['hyperparam_config'], kwargs['problem_config'], kwargs['scale_rewards'],
    #                        kwargs['x_bins_per_dim'], kwargs['u_bins_per_dim'], kwargs['num_GP_points'],
    #                        kwargs['num_KDE_points'], kwargs['num_agents'], kwargs['id'],
    #                        )
    if 'obs_bins_per_dim' in kwargs.keys():
        kwargs['exp_dir'] += f"_{kwargs['obs_bins_per_dim']}"
    if 'squash_to_zero' in kwargs.keys():
        kwargs['exp_dir'] += f"_{kwargs['squash_to_zero']}"
    if 'u_low' in kwargs.keys():
        kwargs['exp_dir'] += f"_ul{kwargs['u_low']}"
    if 'u_high' in kwargs.keys():
        kwargs['exp_dir'] += f"_uh{kwargs['u_high']}"


    if kwargs['game'] == 'comms-two':
        kwargs['game'] = CommsTwoGaussiansEnv
    elif kwargs['game'] == 'major-minor-formation':
        kwargs['game'] = MajorMinorFormationEnv
    elif kwargs['game'] == 'major-minor-potential':
        kwargs['game'] = MajorMinorPotentialFieldEnv
    elif kwargs['game'] == 'major-minor-foraging':
        kwargs['game'] = MajorMinorForagingEnv
    elif kwargs['game'] == 'major-minor-beach-tuple':
        kwargs['game'] = MajorMinorBeachTupleEnv
    else:
        raise NotImplementedError

    if kwargs['solver'] == 'comms':
        kwargs['solver'] = CommsObsGaussianWrapper
    elif kwargs['solver'] == 'major-minor-gaussian':
        kwargs['solver'] = DiscreteGaussianMajorMinorMFCEnv
    elif kwargs['solver'] == 'major-minor-foraging':
        kwargs['solver'] = ForagingMajorMinorMFCEnv
    elif kwargs['solver'] == 'major-minor-comms-gaussian':
        kwargs['solver'] = CommsGaussianMajorMinorMFCEnv
    elif kwargs['solver'] == 'major-minor-pure-discrete-tuple':
        kwargs['solver'] = PureDiscreteTupleMajorMinorMFCEnv
    elif kwargs['solver'] == 'marl-comms':
        kwargs['solver'] = MARLObsWrapperComms
    elif kwargs['solver'] == 'major-minor-marl-comms-separate':
        kwargs['solver'] = MajorMinorMARLObsWrapperCommsSeparate
    elif kwargs['solver'] == 'major-minor-marl-pure-discrete-tuple-separate':
        kwargs['solver'] = MajorMinorMARLObsWrapperDiscretePureTupleSeparate
    elif kwargs['solver'] == 'major-minor-marl-foraging-separate':
        kwargs['solver'] = MajorMinorMARLObsWrapperForagingSeparate
    elif kwargs['solver'] == 'major-minor-marl-discrete-separate':
        kwargs['solver'] = MajorMinorMARLObsWrapperDiscreteSeparate
    else:
        raise NotImplementedError

    if kwargs['policy'] == 'fc':
        kwargs['policy'] = None
    else:
        raise NotImplementedError

    return kwargs


def parse_config():
    args = parse_args()
    return generate_config(args)
