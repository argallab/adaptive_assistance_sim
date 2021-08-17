import mdptoolbox
import numpy as np 
from mdp.mdp_utils import *
import collections

class DiscreteMDP(object):
    def __init__(self, env_params, vs_and_policy_dict=None):
        assert env_params is not None
        assert type(env_params) is collections.OrderedDict
        self.env_params = env_params
        
        self.vs_and_policy_dict = vs_and_policy_dict
        self.rl_algo_type = self.env_params['rl_algo_type']
        self.gamma = self.env_params.get('gamma', 0.98)

        self.rl_algo = None
        self.P = None
        self.R = None
        self.policy = None
        self.value_function = None
        self.Q_function = None
        self.curated_policy_dict = None
        self.state_id_to_state = None
        self.num_states = None

        self._define_mdp()
        print('TOTAL NUM STATES', self.num_states)

        self.task_level_actions = collections.OrderedDict()
        self.action_id_to_task_level_action_map = collections.OrderedDict()
        self.task_level_action_to_action_id_map = collections.OrderedDict()
        self.create_action_dict()
        self.num_actions = len(self.task_level_actions) #discrte actions
        print('TOTAL NUM ACTIONS', self.num_actions)
        print('CREATING TRANSITION MATRIX')
        self._create_transition_matrix()
        print('CREATING REWARD MATRIX')
        self._create_reward_matrix()
        print('SOLVING MDP')
        self._solve_mdp()
    
    def _solve_mdp(self):
        if self.vs_and_policy_dict is None:
            print('Initializing instance of RL solver')
            if self.rl_algo_type == RlAlgoType.ValueIteration:
                self.rl_algo = mdptoolbox.mdp.ValueIteration(self.P, self.R, self.gamma)
            elif self.rl_algo_type == RlAlgoType.PolicyIteration:
                self.rl_algo = mdptoolbox.mdp.PolicyIteration(self.P, self.R, self.gamma)
            elif self.rl_algo_type == RlAlgoType.QLearning:
                self.rl_algo = mdptoolbox.mdp.QLearning(self.P, self.R, self.gamma, n_iter=100000)
            
            print('Running rl algo')
            self.rl_algo.run()
            self.policy = self.rl_algo.policy
            self.value_function = self.rl_algo.V
        else:
            print('Loading V and Policy from file')
            assert 'value_function' in self.vs_and_policy_dict
            assert 'policy' in self.vs_and_policy_dict
            self.value_function = self.vs_and_policy_dict['value_function']
            self.policy = self.vs_and_policy_dict['policy']
            if self.rl_algo_type == RlAlgoType.QLearning:
                self.Q_function = self.vs_and_policy_dict['q_function']

        self._create_curated_policy_dict()
    
    def _create_curated_policy_dict(self):
        raise NotImplementedError

    def _create_state_space(self):
        raise NotImplementedError

    def _create_reward_matrix(self):
        raise NotImplementedError

    def _create_transition_matrix(self):
        raise NotImplementedError

    def create_action_dict(self):
        raise NotImplementedError
    
    #getters
    

    def get_reward_function(self):
        return self.R
    
    def get_transition_function(self):
        return self.P
    
    def get_optimal_policy_for_mdp(self):
        return self.policy, self.curated_policy_dict
    
    def get_random_action(self):
        rand_action_id = np.random.randint(self.num_actions) #random action
        return self.action_id_to_task_level_action_map[rand_action_id]

    def get_value_function(self):
        if self.rl_algo_type == RlAlgoType.QLearning:
            return ((self.value_function), (self.Q_function))
        else:
            return ((self.value_function))

    def get_env_params(self):
        return self.env_params
        
    def get_zero_action(self):
        raise NotImplementedError

    def get_optimal_action(self, state):
        raise NotImplementedError
    
    def get_reward_for_current_transition(self, state, action):
        raise NotImplementedError
    
    def get_random_valid_state(self):
        raise NotImplementedError
    
    def get_next_state_from_state_action(self, state, action):
        raise NotImplementedError

    def get_next_state_from_state_optimal_action(self, state):
        raise NotImplementedError

    def get_optimal_trajectory_from_state(self, state, horizon=100):
        '''
        Unroll the optimal trajectory starting from state. 
        Returns: a list of tuples of (s,a,s')
        '''
        raise NotImplementedError

    def get_random_trajectory_from_state(self, state, horizon=100):
        '''
        Unroll a trajectory starting from state. 
        Returns: a list of tuples of (s,a,s')
        '''
        raise NotImplementedError

    def get_reward_for_current_transition(self, state, action):
        raise NotImplementedError

    