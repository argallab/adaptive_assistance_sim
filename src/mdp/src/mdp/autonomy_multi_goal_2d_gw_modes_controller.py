import numpy as np
from mdp.mdp_discrete_2d_gridworld_with_modes import MDPDiscrete2DGridWorldWithModes
import collections

class AutonomyMultiGoal2DGridWorldModesController(object):
    def __init__(self, env_params, mdp_info_dict=None):
        assert env_params is not None
        assert type(env_params) is collections.OrderedDict
        assert 'grid_width' in env_params
        assert 'grid_height' in env_params
        assert 'grid_obstacles' in env_params
        assert 'grid_goal_states' in env_params
        assert 'sparsity_factor' in env_params
        assert 'rand_direction_factor' in env_params
        assert 'intended_goal_index' in env_params
        assert 'algo_type' in env_params

        self.num_goals = len(env_params['grid_goal_states']) #number of goals in the environment
        self.intended_goal_index = env_params['intended_goal_index'] #intended goal index for autonomy

        #create an indexed list of mdp_discrete_gridworld for each possible goal. Make sure the obstacles are the 'other' goals in the environment
        self.mdp_discrete_grid_worlds_for_each_goal = collections.OrderedDict()
        print('sparsity_factor', env_params['sparsity_factor'])
        print('rand_direction_factor', env_params['rand_direction_factor'])
        print('intended_goal_index', self.intended_goal_index)
        mdp_params_dict = collections.OrderedDict()
        for key,value in env_params.items():
            if key != 'grid_goal_states':
                mdp_params_dict[key] = value

        vs_and_policy_dict = collections.OrderedDict()
        for i in range(self.num_goals):
            if mdp_info_dict is not None:
                vs_and_policy_dict[i] = collections.OrderedDict()
                vs_and_policy_dict[i]['state_value_function'] = mdp_info_dict['state_value_function'][i] #load policy and value function from file. 
                vs_and_policy_dict[i]['action_value_function'] = mdp_info_dict['action_value_function'][i]
                vs_and_policy_dict[i]['policy'] = mdp_info_dict['policy'][i]
            else:
                vs_and_policy_dict[i] = None

        #construct the mdp worlds with proper goal and obstacles 
        for i in range(self.num_goals):
            print('Creating MDP GW for Goal number ', i)
            mdp_params_dict['mdp_goal_state'] = env_params['grid_goal_states'][i]
            goals_that_are_obstacles = [goal for j, goal in enumerate(env_params['grid_goal_states']) if j != i]
            mdp_params_dict['mdp_obstacles'] = env_params['grid_obstacles'] + goals_that_are_obstacles
            
            self.mdp_discrete_grid_worlds_for_each_goal[i] = MDPDiscrete2DGridWorldWithModes(mdp_params_dict, vs_and_policy_dict[i]) #if the value function and policy was precomputed then no need to run the value iteration when instantiating the mdp 

    def random_switch_intended_goal_index(self):
        self.intended_goal_index = np.random.randint(self.num_goals)
        print('New intended goal_index ', self.intended_goal_index)

    def set_intended_goal_index(self, goal_index):
        assert goal_index in list(range(self.num_goals))
        self.intended_goal_index = goal_index

    def get_current_intended_goal_index(self):
        return self.intended_goal_index

    def get_optimal_action_for_intended_goal(self, state, sample_stochastic=False):
        return self.mdp_discrete_grid_worlds_for_each_goal[self.intended_goal_index].get_action_for_state(state, sample_stochastic=sample_stochastic)

    def get_optimal_action_for_goal(self, goal_index, state, sample_stochastic=False):
        return self.mdp_discrete_grid_worlds_for_each_goal[goal_index].get_action_for_state(state, sample_stochastic=sample_stochastic)

    def get_zero_action(self):
        return self.mdp_discrete_grid_worlds_for_each_goal[self.intended_goal_index].get_zero_action()

    def get_mdp_dict_for_all_goals(self):
        return self.mdp_discrete_grid_worlds_for_each_goal