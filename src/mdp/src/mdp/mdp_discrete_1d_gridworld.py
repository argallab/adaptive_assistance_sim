import numpy as np
import collections
import itertools
from mdp.mdp_class import DiscreteMDP
from mdp.mdp_utils import *
from adaptive_assistance_sim_utils import *
import math
from scipy import sparse


class MDPDIscrete1DGridWorld(DiscreteMDP):
    def __init__(self, env_params, vs_and_policy_dict=None):
        super(MDPDIscrete1DGridWorld, self).__init__(env_params, vs_and_policy_dict)

        self.sparsity_factor = env_params.get("sparsity_factor", 0.0)
        self.rand_direction_factor = env_params.get("rand_direction_factor", 0.0)
        self.boltzmann_factor = env_params.get("boltzmann_factor", 100.0)

    def _define_mdp(self):
        assert "grid_width" in self.env_params
        assert "mdp_goal_state" in self.env_params
        assert "robot_type" in self.env_params
        assert "mode_set_type" in self.env_params
        assert "obstacle_penalty" in self.env_params
        assert "step_penalty" in self.env_params
        assert "goal_reward" in self.env_params

        self.width = self.env_params["grid_width"]
        self.goal_state = self.env_params["mdp_goal_state"]  # 1d

        self.obstacle_penalty = self.env_params["obstacle_penalty"]
        self.step_penalty = self.env_params["step_penalty"]
        self.goal_reward = self.env_params["goal_reward"]

        self.num_states = self.width
        self.state_coords = list(range(self.width))
        assert len(self.state_coords) == self.num_states

        self.obstacle_id_list = []
        self.empty_cell_id_list = [s for s in range(self.num_states) if s not in self.obstacle_id_list]
        self.ACTION_VALS = {"move_p": 1, "move_n": -1}

        self.dims = 1  # =3 +1 for mode dimensions

        self.task_level_actions = collections.OrderedDict()
        self.action_id_to_task_level_action_map = collections.OrderedDict()
        self.task_level_action_to_action_id_map = collections.OrderedDict()

    def _create_transition_matrix(self):
        if self.vs_and_policy_dict is not None and "P_matrix" in self.vs_and_policy_dict:
            print("Loading P matrix from file")
            self.P = self.vs_and_policy_dict["P_matrix"]
            assert len(self.P) == self.num_actions
        else:
            self.P = [None] * self.num_actions
            for action_vector in self.task_level_actions.values():
                action_id = action_vector[0]
                task_level_action = action_vector[1]
                action_weight = action_vector[2]
                T = np.zeros((self.num_states, self.num_states), dtype=np.int)

                for state_coord in self.state_coords:
                    new_state_coord, transition_type = self._transition(state_coord, task_level_action)
                    state_id = self._convert_grid_coords_to_1D_state(state_coord)
                    new_state_id = self._convert_grid_coords_to_1D_state(new_state_coord)
                    T[state_id, new_state_id] = 1

                self.P[action_id] = sparse.csr_matrix(T)

    def _create_reward_matrix(self):
        if self.vs_and_policy_dict is not None and "R_matrix" in self.vs_and_policy_dict:
            print("Loading R matrix from file")
            self.R = self.vs_and_policy_dict["R_matrix"]
            assert len(self.R) == self.num_actions
        else:
            self.R = [None] * self.num_actions
            for action_vector in self.task_level_actions.values():
                action_id = action_vector[0]
                task_level_action = action_vector[1]
                action_weight = action_vector[2]
                action_goal_weight = action_vector[3]

                R = np.zeros((self.num_states, self.num_states), dtype=np.int)

                for state_coord in self.state_coords:
                    new_state_coord, transition_type = self._transition(state_coord, task_level_action)
                    state_id = self._convert_grid_coords_to_1D_state(state_coord)
                    new_state_id = self._convert_grid_coords_to_1D_state(new_state_coord)
                    if transition_type == TransitionType.VALID and self.check_if_state_coord_is_goal_state(
                        new_state_coord
                    ):
                        R[state_id, new_state_id] = self.goal_reward * action_goal_weight
                    elif (
                        transition_type == TransitionType.VALID
                    ):  # valid transition into adjacent valid state that is not the goal state
                        R[state_id, new_state_id] = self.step_penalty * action_weight
                    elif transition_type == TransitionType.INTO_WALL:
                        R[state_id, new_state_id] = self.obstacle_penalty

    def _transition(self, state_coord, task_level_action):
        vel_tuple = self._remapped_task_level_action(task_level_action, state_coord)
        if vel_tuple != (0):
            new_state_x = state_coord[Dim.X.value]
            new_state_x = new_state_x + vel_tuple[Dim.X.value]
            new_state_coord = [new_state_x]
            transition_type = TransitionType.VALID
            if new_state_coord[Dim.X.value] < 0 or new_state_coord[Dim.X.value] > self.width - 1:
                transition_type = TransitionType.INTO_WALL

            new_state_coord = self._constrain_within_bounds(new_state_coord)

        return new_state_coord, transition_type

    #
    def _remapped_task_level_action(self, task_level_action, state_coord):
        # convert task-level action to state-dependent control action
        action_vector = [0] * self.dims
        action_val = self.ACTION_VALS[task_level_action]  #
        if task_level_action == "move_p" or task_level_action == "move_n":
            # apply action in the mode that allow movement
            action_vector[0] = action_val
        return tuple(action_vector)

    def _create_transition_matrix(self):
        if self.vs_and_policy_dict is not None and "P_matrix" in self.vs_and_policy_dict:
            print("Loading P matrix from file")
            self.P = self.vs_and_policy_dict["P_matrix"]
            assert len(self.P) == self.num_actions
        else:
            self.P = [None] * self.num_actions

    def check_if_state_coord_is_goal_state(self, state_coord):
        if (state_coord[Dim.X.value]) == self.goal_state:
            return True
        else:
            return False

    def _get_grid_loc_from_state_coord(self, state_coord):
        return tuple([state_coord[Dim.X.value]])

    def _constrain_within_bounds(self, state_coord):
        # make sure the robot doesn't go out of bounds
        state_coord[Dim.X.value] = max(0, min(state_coord[Dim.X.value], self.width - 1))
        return state_coord

    def create_action_dict(self):
        # one D task level actions
        self.task_level_actions["move_p"] = (0, "move_p", 1, 1)
        self.task_level_actions["move_n"] = (1, "move_n", 1, 1)
        self.num_actions = len(self.task_level_actions)

        self.action_id_to_task_level_action_map = {v[0]: v[1] for k, v in self.task_level_actions.items()}
        self.task_level_action_to_action_id_map = {v[1]: v[0] for k, v in self.task_level_actions.items()}
        # self.action_to_modes_that_allow_action = {v[1]:v[2] for k, v in self.task_level_actions.items()}

    def _create_curated_policy_dict(self):
        self.curated_policy_dict = collections.OrderedDict()
        assert len(self.rl_algo.policy) == self.num_states
        for s in range(self.num_states):
            state_coord = self._convert_1D_state_to_grid_coords(s)
            self.curated_policy_dict[state_coord] = self.rl_algo.policy[s]

    def _convert_grid_coords_to_1D_state(self, coord):  # coord can be a tuple, list or np.array
        x_coord = coord[Dim.X.value]
        state_id = x_coord
        return state_id  # scalar

    def _convert_1D_state_to_grid_coords(self, state):
        assert state >= 0 and state < self.num_states
        coord = [0]
        coord[Dim.X.value] = state
        return tuple(coord)

    def get_optimal_action(self, state_coord, return_optimal=True):
        "returns optimal task-level action"
        if self.check_if_state_coord_is_goal_state(state_coord):
            return self.get_zero_action()
        else:
            s = np.random.rand()
            if s < self.sparsity_factor and not return_optimal:
                return self.get_zero_action()
            else:
                d = np.random.rand()
                if d < self.rand_direction_factor and not return_optimal:
                    return self.get_random_action()
                else:
                    state_id = self._convert_grid_coords_to_1D_state(state_coord)
                    action_id = self.rl_algo.policy[state_id]
                    return self.action_id_to_task_level_action_map[action_id]  # movep, moven, mode_l, mode_r

    def get_zero_action(self):
        return "None"

    def get_random_action(self):
        rand_action_id = np.random.randint(self.num_actions)  # random action
        return self.action_id_to_task_level_action_map[rand_action_id]

    def get_empty_cell_id_list(self):
        return self.empty_cell_id_list

    def get_random_valid_state(self):
        rand_state_id = self.empty_cell_id_list[np.random.randint(len(self.empty_cell_id_list))]  # scalar state id
        return self._convert_1D_state_to_grid_coords(rand_state_id)  # tuple (x,y, t mode)

    def get_goal_state(self):
        return self.goal_state

    def get_location(self, state):
        return (state[Dim.X.value], state[Dim.Y.value])

    def get_all_state_coords(self):
        # return the list of all states as coords except obstacles and goals.
        state_coord_list = []
        for state_id in self.empty_cell_id_list:
            state_coord = self._convert_1D_state_to_grid_coords(state_id)
            if self.check_if_state_coord_is_goal_state(state_coord):
                continue
            else:
                state_coord_list.append(state_coord)

        return state_coord_list

    def get_all_state_coords_with_grid_locs_diff_from_goals_and_obs(self):
        state_coord_list = []
        for state_id in self.empty_cell_id_list:
            state_coord = self._convert_1D_state_to_grid_coords(state_id)
            grid_loc_for_state = self._get_grid_loc_from_state_coord(state_coord)  # tuple
            # if the grid loc of state matches the grid loc of goal state skip.
            if grid_loc_for_state == self.get_goal_state()[0]:  # todo change '3' into a Enum var
                continue
            else:
                state_coord_list.append(state_coord)

        return state_coord_list

    def get_next_state_from_state_action(self, state, task_level_action):
        # state is a 1d tuple (x)
        # action is string which is in [movep, moven]
        if task_level_action != "None":
            next_state, _ = self._transition(state, task_level_action)  # np array
            return tuple(next_state)  # make list into tuple (x')
        else:
            return tuple(state)

    def get_optimal_trajectory_from_state(self, state, horizon=100, return_optimal=True):
        # state is 4d tuple (x,y, z, mode)
        sas_trajectory = []
        current_state = state
        for t in range(horizon):
            optimal_action = self.get_optimal_action(current_state, return_optimal=return_optimal)
            next_state, _ = tuple(self._transition(current_state, optimal_action))
            sas_trajectory.append((current_state, optimal_action, next_state))
            if self.check_if_state_coord_is_goal_state(next_state):
                break
            current_state = next_state

        return sas_trajectory

    def get_random_trajectory_from_state(self, state, horizon=100):
        # state is 4d tuple (x,y, z, mode)
        sas_trajectory = []
        current_state = state
        for t in range(horizon):
            current_random_action = self.get_random_action()
            next_state, _ = tuple(self._transition(current_state, current_random_action))
            sas_trajectory.append((current_state, current_random_action, next_state))
            if self.check_if_state_coord_is_goal_state(next_state):
                break
            current_state = next_state

        return sas_trajectory

    def get_prob_a_given_s(self, state_coord, task_level_action):
        assert task_level_action in self.action_id_to_task_level_action_map.values()
        state_id = self._convert_grid_coords_to_1D_state(state_coord)
        action_id = self.task_level_action_to_action_id_map[task_level_action]
        if self.rl_algo_type == RlAlgoType.QLearning:
            p_vec = np.exp(self.boltzmann_factor * self.rl_algo.Q[state_id, :]) / np.sum(
                np.exp(self.boltzmann_factor * self.rl_algo.Q[state_id, :])
            )
            return p_vec[action_id]  # probability associated with action
        else:
            # use deterministic policy
            if self.rl_algo.policy[state_id] == action_id:
                return 1 - self.rand_direction_factor + self.rand_direction_factor / self.num_actions
            else:
                return self.rand_direction_factor / self.num_actions
