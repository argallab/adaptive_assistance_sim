import numpy as np
import collections
import itertools
from mdp.mdp_class import DiscreteMDP
from mdp.mdp_utils import *
import math
from scipy import sparse


class MDPDiscrete2DGridWorldWithModes(DiscreteMDP):
    def __init__(self, env_params, vs_and_policy_dict=None):
        super(MDPDiscrete2DGridWorldWithModes, self).__init__(env_params, vs_and_policy_dict)

        self.sparsity_factor = env_params.get("sparsity_factor", 0.0)
        self.rand_direction_factor = env_params.get("rand_direction_factor", 0.0)
        self.boltzmann_factor = env_params.get("boltzmann_factor", 100.0)

    def _define_mdp(self):
        assert "grid_width" in self.env_params
        assert "grid_height" in self.env_params
        assert "mdp_obstacles" in self.env_params
        assert "mdp_goal_state" in self.env_params
        assert "robot_type" in self.env_params
        assert "mode_set_type" in self.env_params
        assert "obstacle_penalty" in self.env_params
        assert "step_penalty" in self.env_params
        assert "goal_reward" in self.env_params

        self.width = self.env_params["grid_width"]
        self.height = self.env_params["grid_height"]
        # list of 2d. If 2d state, then implies, all possible modes fro the location

        self.obstacles = self.env_params["mdp_obstacles"]
        self.goal_state = self.env_params["mdp_goal_state"]  # 2d
        self.robot_type = self.env_params["robot_type"]
        self.mode_set_type = self.env_params["mode_set_type"]
        self.mode_set = CARTESIAN_MODE_SET_OPTIONS[self.robot_type][self.mode_set_type]
        self.num_modes = len(self.mode_set)  # 1 or 2
        self.env_params["num_modes"] = self.num_modes

        self.obstacle_penalty = self.env_params["obstacle_penalty"]
        self.step_penalty = self.env_params["step_penalty"]
        self.goal_reward = self.env_params["goal_reward"]

        self.num_states = self.width * self.height * self.num_modes
        self.state_coords = list(itertools.product(range(self.width), range(self.height), self.mode_set.keys()))
        assert len(self.state_coords) == self.num_states

        self.obstacle_id_list = []
        for obs in self.obstacles:
            for mode in CARTESIAN_MODE_SET_OPTIONS[self.robot_type][self.mode_set_type].keys():
                obs_tuple = (obs[0], obs[1], mode)
                obs_id = self._convert_grid_coords_to_1D_state(obs_tuple)
                self.obstacle_id_list.append(obs_id)

        self.empty_cell_id_list = [s for s in range(self.num_states) if s not in self.obstacle_id_list]
        self.ACTION_VALS = {"move_p": 1, "move_n": -1, "to_mode_r": 1, "to_mode_l": -1}
        self.dims = CartesianRobotType.R2.value + 1  # =3 +1 for mode dimensions

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

                for mode in CARTESIAN_MODE_SET_OPTIONS[self.robot_type][self.mode_set_type].keys():
                    goal_state_tuple = (self.goal_state[0], self.goal_state[1], mode)
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

                for mode in CARTESIAN_MODE_SET_OPTIONS[self.robot_type][self.mode_set_type].keys():
                    goal_state_tuple = (self.goal_state[0], self.goal_state[1], mode)
                    goal_state_id = self._convert_grid_coords_to_1D_state(goal_state_tuple)
                    R[goal_state_id, :] = 0
                    R[goal_state_id, goal_state_id] = self.goal_reward

                self.R[action_id] = sparse.csr_matrix(R)

            del R

    # HELPER FUNCTIONS

    def _transition(self, state_coord, task_level_action):
        if self._check_in_obstacle(state_coord):
            return state_coord, TransitionType.INVALID
        else:
            remapped_action_tuple = self._remapped_task_level_action(task_level_action, state_coord)
            (vel_tuple, mode_switch_command) = remapped_action_tuple
            if vel_tuple != (0, 0):
                assert mode_switch_command is None

                new_state_x = state_coord[Dim.X.value]
                new_state_y = state_coord[Dim.Y.value]
                new_state_x = new_state_x + vel_tuple[Dim.X.value]
                new_state_y = new_state_y + vel_tuple[Dim.Y.value]
                new_state_coord = [new_state_x, new_state_y, state_coord[Dim.Mode2D.value]]
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
                    new_state_coord = [new_state_x, new_state_y, state_coord[Dim.Mode2D.value]]
            else:
                assert mode_switch_command is not None
                current_mode = state_coord[Dim.Mode2D.value]
                new_state_coord = [
                    state_coord[Dim.X.value],
                    state_coord[Dim.Y.value],
                    self._get_mode_transition(current_mode, mode_switch_command),
                ]
                transition_type = TransitionType.VALID

            return new_state_coord, transition_type

    def _remapped_task_level_action(self, task_level_action, state_coord):
        # convert task-level action to state-dependent control action
        action_vector = [0] * self.dims
        action_val = self.ACTION_VALS[task_level_action]  #
        if task_level_action == "move_p" or task_level_action == "move_n":
            # apply action in the mode that allow movement
            action_vector[state_coord[Dim.Mode2D.value] - 1] = action_val
        elif task_level_action == "to_mode_r" or task_level_action == "to_mode_l":
            action_vector[-1] = action_val
        remapped_action_tuple = self._create_action_tuple(action_vector, state_coord)
        return remapped_action_tuple

    def _create_action_tuple(self, action_vector, state_coord):
        vel_tuple = tuple(action_vector[: CartesianRobotType.R2.value])
        mode_change_action = action_vector[-1]
        if mode_change_action == 0:
            mode_switch_command = None
        else:
            current_mode = state_coord[Dim.Mode2D.value]
            target_mode = current_mode + mode_change_action
            if target_mode == 0:
                target_mode = Dim.Mode2D.value
            if target_mode == Dim.Mode2D.value + 1:
                target_mode = 1

            mode_switch_command = "to" + str(target_mode)

        return (vel_tuple, mode_switch_command)

    def _get_mode_transition(self, current_mode, mode_switch_command):
        if mode_switch_command == "to1":
            new_mode = 1
        elif mode_switch_command == "to2":
            new_mode = 2
        elif mode_switch_command == None:
            new_mode = current_mode

        return new_mode

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

    def create_action_dict(self):
        # one D task level actions
        self.task_level_actions["move_p"] = (0, "move_p", 1, 1)
        self.task_level_actions["move_n"] = (1, "move_n", 1, 1)
        # penalize mode switch more than real motion, therefore the action weight is 10
        self.task_level_actions["to_mode_r"] = (2, "to_mode_r", 10, 1)
        self.task_level_actions["to_mode_l"] = (3, "to_mode_l", 10, 1)

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
        y_coord = coord[Dim.Y.value]
        mode = coord[Dim.Mode2D.value]
        assert mode > 0
        # mode is NOT 0 indexed. Hence the - 1
        state_id = (x_coord * self.height + y_coord) * self.num_modes + (mode - 1)
        return state_id  # scalar

    def _convert_1D_state_to_grid_coords(self, state):
        assert state >= 0 and state < self.num_states
        coord = [0, 0, 0]
        coord[Dim.Mode2D.value] = (state % self.num_modes) + 1
        coord[Dim.Y.value] = (state / self.num_modes) % self.height
        coord[Dim.X.value] = (state / self.num_modes) / self.height
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

    def get_next_state_from_state_action(self, state, task_level_action):
        # state is a 3d tuple (x,y, mode)
        # task_level_action is string which is in [movep, moven, mode_r, mode_l]
        next_state_coord, _ = self._transition(state, task_level_action)  # np array
        return tuple(next_state_coord)  # make list into tuple (x',y',mode')

    def get_optimal_trajectory_from_state(self, state_coord, horizon=100, return_optimal=True):
        # state is 3d tuple (x,y,mode)
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
            next_state_coord, _ = tuple(self._transition(current_state_coord, current_random_action))
            sas_trajectory.append((current_state_coord, current_random_action, next_state_coord))
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
