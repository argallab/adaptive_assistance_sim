import numpy as np
import collections
import itertools
from mdp.mdp_class import DiscreteMDP
from mdp.mdp_utils import *
import os
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "simulators", "scripts"))

from adaptive_assistance_sim_utils import *
import math
from scipy import sparse


class MDPDiscrete2DGridWorld(DiscreteMDP):
    def __init__(self, env_params, vs_and_policy_dict=None):
        super(MDPDiscrete2DGridWorld, self).__init__(env_params, vs_and_policy_dict)

        self.sparsity_factor = env_params.get("sparsity_factor", 0.0)
        self.rand_direction_factor = env_params.get("rand_direction_factor", 0.0)
        self.boltzmann_factor = env_params.get("boltzmann_factor", 100.0)

    def _define_mdp(self):
        assert "grid_width" in self.env_params
        assert "grid_height" in self.env_params
        assert "mdp_obstacles" in self.env_params
        assert "mdp_goal_state" in self.env_params
        assert "robot_type" in self.env_params
        assert "obstacle_penalty" in self.env_params
        assert "step_penalty" in self.env_params
        assert "goal_reward" in self.env_params

        self.width = self.env_params["grid_width"]
        self.height = self.env_params["grid_height"]

        self.obstacles = self.env_params["mdp_obstacles"]
        self.goal_state = self.env_params["mdp_goal_state"]  # 2d
        self.robot_type = self.env_params["robot_type"]

        self.obstacle_penalty = self.env_params["obstacle_penalty"]
        self.step_penalty = self.env_params["step_penalty"]
        self.goal_reward = self.env_params["goal_reward"]

        self.num_states = self.width * self.height
        self.state_coords = list(itertools.product(range(self.width), range(self.height)))
        assert len(self.state_coords) == self.num_states

        self.obstacle_id_list = []
        for obs in self.obstacles:
            obs_tuple = (obs[0], obs[1])
            obs_id = self._convert_grid_coords_to_1D_state(obs_tuple)
            self.obstacle_id_list.append(obs_id)

        self.empty_cell_id_list = [s for s in range(self.num_states) if s not in self.obstacle_id_list]
        self.dims = CartesianRobotType.R2.value

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
                action_id = action_vector[0]  # number
                task_level_action = action_vector[1]  # string
                action_weight = action_vector[2]  # weight

                T = np.zeros((self.num_states, self.num_states), dtype=np.int)
                for state_coord in self.state_coords:
                    new_state_coord, transition_type = self._transition(state_coord, task_level_action)
                    state_id = self._convert_grid_coords_to_1D_state(state_coord)
                    new_state_id = self._convert_grid_coords_to_1D_state(new_state_coord)
                    T[state_id, new_state_id] = 1

                goal_state_tuple = (self.goal_state[0], self.goal_state[1])
                goal_state_id = self._convert_grid_coords_to_1D_state(goal_state_tuple)
                T[goal_state_id, :] = 0
                T[goal_state_id, goal_state_id] = 1

                self.P[action_id] = sparse.csr_matrix(T)

            del T

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
                # TransitionType.INVALID will automatically have 0.
                for state_coord in self.state_coords:
                    new_state_coord, transition_type = self._transition(state_coord, task_level_action)
                    state_id = self._convert_grid_coords_to_1D_state(state_coord)
                    new_state_id = self._convert_grid_coords_to_1D_state(new_state_coord)
                    if transition_type == TransitionType.INTO_OBSTACLE:
                        R[state_id, new_state_id] = self.obstacle_penalty
                        # its a valid transition into goal
                    elif transition_type == TransitionType.VALID and self.check_if_state_coord_is_goal_state(
                        new_state_coord
                    ):
                        R[state_id, new_state_id] = self.goal_reward * action_goal_weight
                    elif (
                        transition_type == TransitionType.VALID
                    ):  # valid transition into adjacent valid state that is not the goal state
                        R[state_id, new_state_id] = self.step_penalty * action_weight
                    elif transition_type == TransitionType.INTO_WALL:
                        R[state_id, new_state_id] = self.obstacle_penalty

                goal_state_tuple = (self.goal_state[0], self.goal_state[1])
                goal_state_id = self._convert_grid_coords_to_1D_state(goal_state_tuple)
                R[goal_state_id, :] = 0
                R[goal_state_id, goal_state_id] = self.goal_reward

                self.R[action_id] = sparse.csr_matrix(R)

            del R

    def _transition(self, state_coord, task_level_action):
        if self._check_in_obstacle(state_coord):
            # ran into obstacle invalid. therefore stay put
            return state_coord, TransitionType.INVALID
        else:
            vel_tuple = self._remapped_task_level_action(task_level_action, state_coord)
            new_state_x = state_coord[Dim.X.value]
            new_state_y = state_coord[Dim.Y.value]
            new_state_x = new_state_x + vel_tuple[Dim.X.value]
            new_state_y = new_state_y + vel_tuple[Dim.Y.value]
            new_state_coord = [new_state_x, new_state_y]
            transition_type = TransitionType.VALID
            if (
                new_state_coord[Dim.X.value] < 0
                or new_state_coord[Dim.X.value] > self.width - 1
                or new_state_coord[Dim.Y.value] < 0
                or new_state_coord[Dim.Y.value] > self.height - 1
            ):
                transition_type = TransitionType.INTO_WALL
            new_state_coord = self._constrain_within_bounds(new_state_coord)
            if self._check_in_obstacle(new_state_coord):
                transition_type = TransitionType.INTO_OBSTACLE
                new_state_x = state_coord[Dim.X.value]
                new_state_y = state_coord[Dim.Y.value]
                new_state_coord = [new_state_x, new_state_y]

            return new_state_coord, transition_type

    def _remapped_task_level_action(self, task_level_action, state_coord):
        action_vector = [0] * self.dims

        if task_level_action == "move_up":
            action_vector = [0, 1]
        elif task_level_action == "move_down":
            action_vector = [0, -1]
        elif task_level_action == "move_left":
            action_vector = [-1, 0]
        elif task_level_action == "move_right":
            action_vector = [1, 0]
        elif task_level_action == "move_up_right":
            action_vector = [1, 1]
        elif task_level_action == "move_up_left":
            action_vector = [-1, 1]
        elif task_level_action == "move_down_left":
            action_vector = [-1, -1]
        elif task_level_action == "move_down_right":
            action = [1, -1]

        return action_vector

    def _check_in_obstacle(self, state_coord):
        if self._get_grid_loc_from_state_coord(state_coord) in self.obstacles:
            return True
        else:
            return False
    
    def check_if_state_coord_is_goal_state(self, state_coord):
        if (state_coord[Dim.X.value], state_coord[Dim.Y.value]) == self.goal_state:
            return True
        else:
            return False

    def _get_grid_loc_from_state_coord(self, state_coord):
        return tuple([state_coord[Dim.X.value], state_coord[Dim.Y.value]])

    def _constrain_within_bounds(self, state_coord):
        # make sure the robot doesn't go out of bounds
        state_coord[Dim.X.value] = max(0, min(state_coord[Dim.X.value], self.width - 1))
        state_coord[Dim.Y.value] = max(0, min(state_coord[Dim.Y.value], self.height - 1))
        return state_coord

    def _convert_grid_coords_to_1D_state(self, coord):  # coord can be a tuple, list or np.array
        x_coord = coord[Dim.X.value]
        y_coord = coord[Dim.Y.value]
        state_id = x_coord * self.height + y_coord
        return state_id  # scalar

    def _convert_1D_state_to_grid_coords(self, state):
        assert state >= 0 and state < self.num_states
        coord = [0, 0]
        coord[Dim.Y.value] = (state) % self.height
        coord[Dim.X.value] = (state) // self.height
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
        # zero task level action
        return "None"

    def get_random_action(self):
        rand_action_id = np.random.randint(self.num_actions)  # random action
        return self.action_id_to_task_level_action_map[rand_action_id]

    def get_empty_cell_id_list(self):
        return self.empty_cell_id_list

    def get_random_valid_state(self, is_not_goal=False):
        rand_state_id = self.empty_cell_id_list[np.random.randint(len(self.empty_cell_id_list))]  # scalar state id
        state_coord = self._convert_1D_state_to_grid_coords(rand_state_id)
        if is_not_goal:
            while self.check_if_state_coord_is_goal_state(state_coord):
                # scalar state id
                rand_state_id = self.empty_cell_id_list[np.random.randint(len(self.empty_cell_id_list))]
                state_coord = self._convert_1D_state_to_grid_coords(rand_state_id)
        else:
            rand_state_id = self.empty_cell_id_list[np.random.randint(len(self.empty_cell_id_list))]
            state_coord = self._convert_1D_state_to_grid_coords(rand_state_id)

        return state_coord  # tuple (x,y)

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
            if grid_loc_for_state == self.get_goal_state()[0:2]:
                continue
            else:
                state_coord_list.append(state_coord)

        return state_coord_list

    def get_next_state_from_state_action(self, state, task_level_action):
        # state is a 3d tuple (x,y)
        # task_level_action is string which is in [move_up, move_down....]
        if not task_level_action == "None":
            next_state_coord, _ = self._transition(state, task_level_action)  # np array
            return tuple(next_state_coord)  # make list into tuple (x',y')
        else:
            return tuple(state)

    def get_optimal_trajectory_from_state(self, state_coord, horizon=100, return_optimal=True):
        # state is 3d tuple (x,y)
        sas_trajectory = []
        current_state_coord = state_coord
        for t in range(horizon):
            optimal_action = self.get_optimal_action(current_state_coord, return_optimal=return_optimal)
            next_state_coord, _ = tuple(self._transition(current_state_coord, optimal_action))
            sas_trajectory.append((current_state_coord, optimal_action, next_state_coord))
            if self.check_if_state_coord_is_goal_state(next_state_coord):
                break
            current_state_coord = next_state_coord

        return sas_trajectory

    def get_random_trajectory_from_state(self, state_coord, horizon=100):
        sas_trajectory = []
        current_state = state_coord
        for t in range(horizon):
            current_random_action = self.get_random_action()
            next_state_coord, _ = tuple(self._transition(current_state, current_random_action))
            sas_trajectory.append((current_state, current_random_action, next_state_coord))
            if self.check_if_state_coord_is_goal_state(next_state_coord):
                break
            current_state = next_state_coord

        return sas_trajectory

    def get_reward_for_current_transition(self, state_coord, task_level_action):
        # state is a tuple.
        # action is a tuple or list. Ideally, tuple for consistency
        if task_level_action != "None":
            action_id = self.task_level_action_to_action_id_map[tuple(task_level_action)]
            state_id = self._convert_grid_coords_to_1D_state(state_coord)
            new_state_coord, _ = self._transition(state_coord, task_level_action)
            new_state_id = self._convert_grid_coords_to_1D_state(new_state_coord)
            return self.R[action_id][state_id, new_state_id]
        else:
            return 0.0

    def get_action_to_action_id_map(self):
        return self.task_level_action_to_action_id_map

    def get_prob_a_given_s(self, state_coord, task_level_action):
        assert task_level_action in self.action_id_to_task_level_action_map.values()
        state_id = self._convert_grid_coords_to_1D_state(state_coord)
        action_id = self.task_level_action_to_action_id_map[task_level_action]
        if self.rl_algo_type == RlAlgoType.QLearning:
            pass
        else:
            if self.rl_algo.policy[state_id] == action_id:
                return 1 - self.rand_direction_factor + self.rand_direction_factor / self.num_actions
            else:
                return self.rand_direction_factor / self.num_actions

    def _create_curated_policy_dict(self):
        self.curated_policy_dict = collections.OrderedDict()
        assert len(self.rl_algo.policy) == self.num_states
        for s in range(self.num_states):
            state_coord = self._convert_1D_state_to_grid_coords(s)
            self.curated_policy_dict[state_coord] = self.rl_algo.policy[s]

    def create_action_dict(self):
        # one D task level actions
        self.task_level_actions["move_up"] = (0, "move_up", 1, 1)
        self.task_level_actions["move_up_left"] = (1, "move_up_left", np.sqrt(2), 1)
        self.task_level_actions["move_left"] = (2, "move_left", 1, 1)
        self.task_level_actions["move_down_left"] = (3, "move_down_left", np.sqrt(2), 1)
        self.task_level_actions["move_down"] = (4, "move_down", 1, 1)
        self.task_level_actions["move_down_right"] = (5, "move_down_right", np.sqrt(2), 1)
        self.task_level_actions["move_right"] = (6, "move_right", 1, 1)
        self.task_level_actions["move_up_right"] = (7, "move_up_right", np.sqrt(2), 1)

        self.num_actions = len(self.task_level_actions)

        self.action_id_to_task_level_action_map = {v[0]: v[1] for k, v in self.task_level_actions.items()}
        self.task_level_action_to_action_id_map = {v[1]: v[0] for k, v in self.task_level_actions.items()}
