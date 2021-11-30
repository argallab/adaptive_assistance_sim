from os import stat
import numpy as np
import collections
import itertools
from mdp.mdp_class import DiscreteMDP

from mdp.mdp_utils import *
from adaptive_assistance_sim_utils import *
import math
from scipy import sparse


class MDPDiscrete3DGridWorldWithModes(DiscreteMDP):
    def __init__(self, env_params, vs_and_policy_dict=None):
        super(MDPDiscrete3DGridWorldWithModes, self).__init__(env_params, vs_and_policy_dict)

        self.sparsity_factor = env_params.get("sparsity_factor", 0.0)
        self.rand_direction_factor = env_params.get("rand_direction_factor", 0.0)
        self.boltzmann_factor = env_params.get("boltzmann_factor", 100.0)

    def _define_mdp(self):
        assert "grid_width" in self.env_params  # along l-r (x)
        assert "grid_depth" in self.env_params  # along (f-b) (y)
        assert "grid_height" in self.env_params  # along u-d (z)

        assert "mdp_obstacles" in self.env_params
        assert "mdp_goal_state" in self.env_params  # 3D

        # assert "num_discrete_orientations" in self.env_params
        assert "robot_type" in self.env_params  # R3 (x, y, z)
        assert "mode_set_type" in self.env_params  # 1D for HA
        assert "obstacle_penalty" in self.env_params
        assert "step_penalty" in self.env_params
        assert "goal_reward" in self.env_params

        self.width = self.env_params["grid_width"]
        self.depth = self.env_params["grid_depth"]
        self.height = self.env_params["grid_height"]

        # list of 3d. If 3d state, then implies, all possible modes ares obstacle for the 3d location
        self.obstacles = self.env_params["mdp_obstacles"]
        self.goal_state = self.env_params["mdp_goal_state"]  # 3d (x, y, z)
        self.robot_type = self.env_params["robot_type"]  # R3

        # assert this is one D. As the task level actions are for 1D interfaces.
        self.mode_set_type = self.env_params["mode_set_type"]
        self.mode_set = CARTESIAN_MODE_SET_OPTIONS[self.robot_type][self.mode_set_type]
        self.num_modes = len(self.mode_set)  # Three 1D modes corresponding to x, y, z

        self.obstacle_penalty = self.env_params["obstacle_penalty"]
        self.step_penalty = self.env_params["step_penalty"]
        self.goal_reward = self.env_params["goal_reward"]

        self.num_states = self.width * self.depth * self.height * self.num_modes
        self.state_coords = list(
            itertools.product(range(self.width), range(self.depth), range(self.height), self.mode_set.keys())
        )
        assert len(self.state_coords) == self.num_states
        self.obstacle_id_list = []
        for obs in self.obstacles:
            for mode in self.mode_set.keys():
                obs_tuple = (obs[0], obs[1], obs[2], mode)
                obs_id = self._convert_grid_coords_to_1D_state(obs_tuple)
                self.obstacle_id_list.append(obs_id)

        self.empty_cell_id_list = [s for s in range(self.num_states) if s not in self.obstacle_id_list]
        self.ACTION_VALS = {"move_p": 1, "move_n": -1, "to_mode_r": 1, "to_mode_l": -1}
        self.dims = CartesianRobotType.R3.value + 1  # =4 +1 for mode dimensions

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

                # it doesn't matter what the final mode is for the goal state.
                for mode in CARTESIAN_MODE_SET_OPTIONS[self.robot_type][self.mode_set_type].keys():
                    # make (gx, gy, gz, m) for all m
                    goal_state_tuple = (
                        self.goal_state[Dim.X.value],
                        self.goal_state[Dim.Y.value],
                        self.goal_state[Dim.Z.value],
                        mode,
                    )
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
                R = np.zeros((self.num_states, self.num_states), dtype=np.int)  # all INVALID transitions will have 0.
                for state_coord in self.state_coords:
                    new_state_coord, transition_type = self._transition(state_coord, task_level_action)
                    state_id = self._convert_grid_coords_to_1D_state(state_coord)
                    new_state_id = self._convert_grid_coords_to_1D_state(new_state_coord)
                    if transition_type == TransitionType.INTO_OBSTACLE:
                        R[state_id, new_state_id] = self.obstacle_penalty
                    elif transition_type == TransitionType.VALID and self._check_if_state_coord_is_goal_state(
                        new_state_coord
                    ):  # its a valid transition into goal
                        R[state_id, new_state_id] = self.goal_reward * action_goal_weight
                    elif transition_type == TransitionType.VALID:
                        R[state_id, new_state_id] = self.step_penalty * action_weight
                    elif transition_type == TransitionType.INTO_WALL:
                        R[state_id, new_state_id] = self.obstacle_penalty

                # it doesn't matter what the final mode is for the goal state.
                for mode in CARTESIAN_MODE_SET_OPTIONS[self.robot_type][self.mode_set_type].keys():
                    goal_state_tuple = (
                        self.goal_state[Dim.X.value],
                        self.goal_state[Dim.Y.value],
                        self.goal_state[Dim.Z.value],
                        mode,
                    )  # make (gx, gy, gz, m) for all m
                    goal_state_id = self._convert_grid_coords_to_1D_state(goal_state_tuple)  # ()
                    R[goal_state_id, :] = 0
                    R[goal_state_id, goal_state_id] = self.goal_reward

                self.R[action_id] = sparse.csr_matrix(R)

            del R

    # HELPER function

    def _transition(self, state_coord, task_level_action):
        if self._check_in_obstacle(state_coord):
            return state_coord, TransitionType.INVALID
        else:
            remapped_action_tuple = self._remapped_task_level_action(state_coord, task_level_action)
            (vel_tuple, mode_switch_command) = remapped_action_tuple
            if vel_tuple != (0, 0, 0):
                assert mode_switch_command is None
                new_state_x = state_coord[Dim.X.value]
                new_state_y = state_coord[Dim.Y.value]
                new_state_z = state_coord[Dim.Z.value]
                # "move_p" results in "decrease of coordinates" because we want soft puff to match to rightwards motion of JACO (which is the -ve x direction)
                new_state_x = new_state_x - vel_tuple[Dim.X.value]
                new_state_y = new_state_y - vel_tuple[Dim.Y.value]
                new_state_z = new_state_z - vel_tuple[Dim.Z.value]
                new_state_coord = [new_state_x, new_state_y, new_state_z, state_coord[Dim.Mode3D.value]]
                transition_type = TransitionType.VALID
                if (
                    new_state_coord[Dim.X.value] < 0
                    or new_state_coord[Dim.X.value] > self.width - 1
                    or new_state_coord[Dim.Y.value] < 0
                    or new_state_coord[Dim.Y.value] > self.depth - 1
                    or new_state_coord[Dim.Z.value] < 0
                    or new_state_coord[Dim.Z.value] > self.height - 1
                ):
                    transition_type = TransitionType.INTO_WALL

                new_state_coord = self._constrain_within_bounds(new_state_coord)
                if self._check_in_obstacle(new_state_coord):
                    transition_type = TransitionType.INTO_OBSTACLE
                    new_state_x = state_coord[Dim.X.value]
                    new_state_y = state_coord[Dim.Y.value]
                    new_state_z = state_coord[Dim.Z.value]
                    new_state_coord = [new_state_x, new_state_y, new_state_z, state_coord[Dim.Mode3D.value]]
            else:
                assert mode_switch_command is not None
                current_mode = state_coord[Dim.Mode3D.value]
                new_state_coord = [
                    state_coord[Dim.X.value],
                    state_coord[Dim.Y.value],
                    state_coord[Dim.Z.value],
                    self._get_mode_transition(current_mode, mode_switch_command),
                ]
                transition_type = TransitionType.VALID

            return new_state_coord, transition_type

    def _remapped_task_level_action(self, state_coord, task_level_action):
        # 4d action to decide whether increment or decrement happens along which dimension
        action_vector = [0] * self.dims
        # deal with movement type actions
        if task_level_action == "move_p" or task_level_action == "move_n":
            action_val = self.ACTION_VALS[task_level_action]  # increment of decrement in the mode that allow movement
            action_vector[
                state_coord[Dim.Mode3D.value] - 1
            ] = action_val  # -1 in index because 1,2,3 are the mode values
        elif task_level_action == "to_mode_r" or task_level_action == "to_mode_l":  # mode switch action
            action_val = self.ACTION_VALS[task_level_action]
            action_vector[-1] = action_val  # [0,0,0,-1/+1] -1 in the because mode is the LAST dimension
        remapped_action_tuple = self._create_action_tuple(action_vector, state_coord)
        return remapped_action_tuple  # ((x,y,z), mode_switch)

    def _create_action_tuple(self, action_vector, state_coord):
        vel_tuple = tuple(action_vector[: CartesianRobotType.R3.value])
        mode_change_action = action_vector[-1]  # 0, +1, -1 mode_r or mode_lf
        if mode_change_action == 0:
            mode_switch_command = None
        else:
            current_mode = state_coord[Dim.Mode3D.value]  # ,1,2,3
            target_mode = current_mode + mode_change_action
            # wrap around mode
            if target_mode == 0:
                target_mode = 1
            if target_mode == Dim.Mode3D.value + 1:
                target_mode = Dim.Mode3D.value

            mode_switch_command = "to" + str(target_mode)

        # mode switch command is to1, to2, to3
        return (vel_tuple, mode_switch_command)

    def _get_mode_transition(self, current_mode, mode_switch_command):
        # print(current_mode, mode_switch_command)
        if mode_switch_command == "to1":
            new_mode = 1
        elif mode_switch_command == "to2":
            new_mode = 2
        elif mode_switch_command == "to3":
            new_mode = 3
        elif mode_switch_command == None:
            new_mode = current_mode

        return new_mode

    def _check_in_obstacle(self, state_coord):
        if self._get_grid_loc_from_state_coord(state_coord) in self.obstacles:
            return True
        else:
            return False

    def _get_grid_loc_from_state_coord(self, state_coord):
        return tuple([state_coord[Dim.X.value], state_coord[Dim.Y.value], state_coord[Dim.Z.value]])

    def _check_if_state_coord_is_goal_state(self, state_coord):
        if (state_coord[Dim.X.value], state_coord[Dim.Y.value], state_coord[Dim.Z.value]) == self.goal_state:
            return True
        else:
            return False

    def _convert_grid_coords_to_1D_state(self, coord):
        x_coord = coord[Dim.X.value]
        y_coord = coord[Dim.Y.value]
        z_coord = coord[Dim.Z.value]
        mode = coord[Dim.Mode3D.value]
        state_id = ((x_coord * self.depth + y_coord) * self.height + z_coord) * self.num_modes + mode - 1
        return state_id

    def _convert_1D_state_to_grid_coords(self, state):
        assert state >= 0 and state < self.num_states
        coord = [0, 0, 0, 0]
        coord[Dim.Mode3D.value] = (state % self.num_modes) + 1
        coord[Dim.Z.value] = (state / self.num_modes) % self.height
        coord[Dim.Y.value] = ((state / self.num_modes) / self.height) % self.depth
        coord[Dim.X.value] = (((state / self.num_modes) / self.height)) / self.depth
        return tuple(coord)

    def _constrain_within_bounds(self, state_coord):
        # make sure the robot doesn't go out of bounds
        state_coord[Dim.X.value] = max(0, min(state_coord[Dim.X.value], self.width - 1))
        state_coord[Dim.Y.value] = max(0, min(state_coord[Dim.Y.value], self.depth - 1))
        state_coord[Dim.Z.value] = max(0, min(state_coord[Dim.Z.value], self.height - 1))

        return state_coord

    def create_action_dict(self):
        # action_id, task_level_action_name, reward_weight_each_action, goal_reward_weight_each_action
        self.task_level_actions["move_p"] = (0, "move_p", 1, 1)
        self.task_level_actions["move_n"] = (1, "move_n", 1, 1)
        # penalize mode switch more than real motion, therefore the action weight is 10
        self.task_level_actions["to_mode_r"] = (2, "to_mode_r", 1, 1)
        self.task_level_actions["to_mode_l"] = (3, "to_mode_l", 1, 1)

        self.action_id_to_task_level_action_map = {v[0]: v[1] for k, v in self.task_level_actions.items()}
        self.task_level_action_to_action_id_map = {v[1]: v[0] for k, v in self.task_level_actions.items()}

    def _create_curated_policy_dict(self):
        self.curated_policy_dict = collections.OrderedDict()
        assert len(self.rl_algo.policy) == self.num_states
        for s in range(self.num_states):
            state_coord = self._convert_1D_state_to_grid_coords(s)
            self.curated_policy_dict[state_coord] = self.rl_algo.policy[s]

    def get_optimal_action(self, state_coord, return_optimal=True):
        if self._check_if_state_coord_is_goal_state(state_coord):
            return self.get_zero_action()
        else:
            s = np.random.rand()
            # print(self.sparsity_factor, self.rand_direction_factor)
            if s < self.sparsity_factor and not return_optimal:
                # print('sparse')
                return self.get_zero_action()
            else:
                d = np.random.rand()
                if d < self.rand_direction_factor and not return_optimal:
                    # print('rand')
                    return self.get_random_action()
                else:
                    state_id = self._convert_grid_coords_to_1D_state(state_coord)
                    action_id = self.rl_algo.policy[state_id]
                    return self.action_id_to_task_level_action_map[action_id]  # movep, moven, mode_l, mode_r

    def get_zero_action(self):
        # zero task level action
        return "None"

    def get_random_valid_state(self, is_not_goal=False):
        rand_state_id = self.empty_cell_id_list[np.random.randint(len(self.empty_cell_id_list))]  # scalar state id
        state_coord = self._convert_1D_state_to_grid_coords(rand_state_id)
        if is_not_goal:
            while self._check_if_state_coord_is_goal_state(state_coord):
                # scalar state id
                rand_state_id = self.empty_cell_id_list[np.random.randint(len(self.empty_cell_id_list))]
                state_coord = self._convert_1D_state_to_grid_coords(rand_state_id)

        return state_coord  # tuple (x,y, z, mode)

    def get_goal_state(self):
        return self.goal_state  # (x,y,z)

    def get_location(self, state):
        return (state[Dim.X.value], state[Dim.Y.value], state[Dim.Z.value])  # (x,y, z) position

    def get_all_state_coords(self):
        # return the list of all states as coords except obstacles and goals.
        state_coord_list = []
        for state_id in self.empty_cell_id_list:  # already excludes the obstacles in empty cell_id list
            state_coord = self._convert_1D_state_to_grid_coords(state_id)
            if self._check_if_state_coord_is_goal_state(state_coord):
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
            if grid_loc_for_state == self.get_goal_state()[0:3]:  # todo change '3' into a Enum var
                continue
            else:
                state_coord_list.append(state_coord)

        return state_coord_list

    def get_next_state_from_state_action(self, state, task_level_action):
        # state is a 4d tuple (x,y, z, mode)
        # action is string which is in [movep, moven, mode_r, mode_l]
        if task_level_action != "None":
            next_state, _ = self._transition(state, task_level_action)  # np array
            return tuple(next_state)  # make list into tuple (x',y',z', mode)
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
            if self._check_if_state_coord_is_goal_state(next_state):
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
            if self._check_if_state_coord_is_goal_state(next_state):
                break
            current_state = next_state

        return sas_trajectory

    def get_reward_for_current_transition(self, state, task_level_action):
        # state is a tuple.
        # action is a tuple or list. Ideally, tuple for consistency
        if task_level_action != "None":
            action_id = self.task_level_action_to_action_id_map[tuple(task_level_action)]
            state_id = self._convert_grid_coords_to_1D_state(state)
            new_state, _ = self._transition(state, task_level_action)
            new_state_id = self._convert_grid_coords_to_1D_state(new_state)
            return self.R[action_id][state_id, new_state_id]
        else:
            return 0.0

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
