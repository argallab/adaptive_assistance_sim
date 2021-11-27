#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import collections
import os
from scipy import special
import sys
import rospkg
import copy
from scipy.stats import entropy
import itertools

sys.path.append(os.path.join(rospkg.RosPack().get_path("simulators"), "scripts"))
from generate_adaptive_assistance_trials import create_random_obstacles, create_random_goals, create_random_start_state
from mdp.mdp_discrete_3d_gridworld_with_modes import MDPDiscrete3DGridWorldWithModes
from mdp.mdp_discrete_1d_gridworld import MDPDIscrete1DGridWorld
from disamb_algo.discrete_mi_disamb_algo_3d import DiscreteMIDisambAlgo3D
from adaptive_assistance_sim_utils import *
from mdp.mdp_utils import *
import matplotlib.pyplot as plt

# low level commands issued by the snp interface. hp = hard puff, hs= hard sip, sp = soft puff, ss = soft sip. Also the domain for ui and um
INTERFACE_LEVEL_ACTIONS = ["hp", "hs", "sp", "ss"]
# high level actions, move_p = move in positive direction, move_n = move in negative direction, mode_r = switch mode to right, mode_l = switch mode to left. positive and negative is conditioned on mode
TASK_LEVEL_ACTIONS = ["move_p", "move_n", "to_mode_r", "to_mode_l"]
# true mapping of a to phi
TRUE_TASK_ACTION_TO_INTERFACE_ACTION_MAP = collections.OrderedDict(
    {"move_p": "sp", "move_n": "ss", "to_mode_r": "hp", "to_mode_l": "hs"}
)
# true inverse mapping of phi to a
TRUE_INTERFACE_ACTION_TO_TASK_ACTION_MAP = collections.OrderedDict(
    {v: k for k, v in TRUE_TASK_ACTION_TO_INTERFACE_ACTION_MAP.items()}
)

INTERFACE_LEVEL_ACTIONS_TO_NUMBER_ID = {"sp": 0, "ss": 1, "hp": 2, "hs": 3}

# p(phii|a)
P_PHI_GIVEN_A = collections.OrderedDict()
# Lower the number, lower the error. Between 0 and 1. If 0, the p(ui|a) is delta and same as the true mapping
PHI_GIVEN_A_NOISE = 0.0

# p(phm|ui)
P_PHM_GIVEN_PHI = collections.OrderedDict()
PHM_GIVEN_PHI_NOISE = 0.0  # Lower the number, lower the error. Between 0 and 1. If 0, no difference between ui and um

PHI_SPARSE_LEVEL = 0.0
PHM_SPARSE_LEVEL = 0.0


NUM_GOALS = 4
OCCUPANCY_LEVEL = 0.1
MAX_PATCHES = 4

GRID_WIDTH = 10
GRID_DEPTH = 8
GRID_HEIGHT = 8
ENTROPY_THRESHOLD = 0.6
SPATIAL_WINDOW_HALF_LENGTH = 3

# DICTIONARIES
DIM_TO_MODE_INDEX_XYZ = {"x": 0, "y": 1, "z": 2, "gr": 3}
MODE_INDEX_TO_DIM_XYZ = {v: k for k, v in DIM_TO_MODE_INDEX.items()}


def simulate_trajectories(start_coord, goal_id, mdp_list, horizon=100):
    mdp_g = mdp_list[goal_id]
    opt_traj_from_state = mdp_g.get_optimal_trajectory_from_state(start_coord, horizon=100, return_optimal=True)
    print(opt_traj_from_state)


def create_mdp_list(mdp_env_params):
    mdp_list = []
    for i, g in enumerate(mdp_env_params["all_goals"]):
        mdp_env_params["mdp_goal_state"] = g
        # 3d trans goals.
        goals_that_are_obs = [(g_obs[0], g_obs[1], g_obs[2]) for g_obs in mdp_env_params["all_goals"] if g_obs != g]
        mdp_env_params["mdp_obstacles"] = copy.deepcopy(mdp_env_params["original_mdp_obstacles"])
        mdp_env_params["mdp_obstacles"].extend(goals_that_are_obs)

        discrete_jaco_SE3_mdp = MDPDiscrete3DGridWorldWithModes(copy.deepcopy(mdp_env_params))
        mdp_list.append(discrete_jaco_SE3_mdp)

    return mdp_list


def create_mdp_env_param_dict():
    mdp_env_params = collections.OrderedDict()
    mdp_env_params["rl_algo_type"] = RlAlgoType.ValueIteration
    mdp_env_params["gamma"] = 0.96
    mdp_env_params["grid_width"] = GRID_WIDTH
    mdp_env_params["grid_depth"] = GRID_DEPTH
    mdp_env_params["grid_height"] = GRID_HEIGHT

    # for MDP we are treating JACO as a 3D robot.
    mdp_env_params["robot_type"] = CartesianRobotType.R3
    mdp_env_params["mode_set_type"] = ModeSetType.OneD

    mdp_env_params["original_mdp_obstacles"] = []

    goal_list = [(7, 1, 1), (7, 7, 1), (2, 2, 1), (2, 2, 6)]

    print("MDP GOAL LIST", goal_list)

    mdp_env_params["all_goals"] = goal_list
    mdp_env_params["obstacle_penalty"] = -100
    mdp_env_params["goal_reward"] = 100
    mdp_env_params["step_penalty"] = -10
    mdp_env_params["sparsity_factor"] = 0.0
    mdp_env_params["rand_direction_factor"] = 0.0
    mdp_env_params["mdp_obstacles"] = []

    return mdp_env_params


def convert_discrete_state_to_continuous_position(discrete_state, cell_size, world_bounds):
    x_coord = discrete_state[0]
    y_coord = discrete_state[1]
    z_coord = discrete_state[2]
    mode = discrete_state[3] - 1

    robot_position = [
        x_coord * cell_size["x"] + cell_size["x"] / 2.0 + world_bounds["xrange"]["lb"],
        y_coord * cell_size["y"] + cell_size["y"] / 2.0 + world_bounds["yrange"]["lb"],
        z_coord * cell_size["z"] + cell_size["z"] / 2.0 + world_bounds["zrange"]["lb"],
    ]

    return robot_position


def create_1d_mdp_env_params_dict(mdp_env_params):
    mdp_dict_1d = collections.defaultdict(list)
    for i, g in enumerate(mdp_env_params["all_goals"]):
        for j, g_coord in enumerate(g):
            mdp_env_params["mdp_goal_state"] = g_coord
            discrete_jaco_1d_mdp = MDPDIscrete1DGridWorld(copy.deepcopy(mdp_env_params))
            mdp_dict_1d[j].append(discrete_jaco_1d_mdp)  # j is the 'mode_id'. g_coord is '1d goal'

    return mdp_dict_1d


if __name__ == "__main__":
    mdp_env_params = create_mdp_env_param_dict()
    mdp_list = create_mdp_list(mdp_env_params)
    mdp_dict_1d = create_1d_mdp_env_params_dict(mdp_env_params)
    world_bounds = collections.OrderedDict()
    world_bounds["xrange"] = collections.OrderedDict()
    world_bounds["yrange"] = collections.OrderedDict()
    world_bounds["zrange"] = collections.OrderedDict()
    world_bounds["xrange"]["lb"] = -0.7
    world_bounds["yrange"]["lb"] = -0.7
    world_bounds["zrange"]["lb"] = 0.0
    world_bounds["xrange"]["ub"] = 0.7
    world_bounds["yrange"]["ub"] = 0.0
    world_bounds["zrange"]["ub"] = 0.6
    mdp_env_params["cell_size"] = collections.OrderedDict()
    mdp_env_params["cell_size"]["x"] = (world_bounds["xrange"]["ub"] - world_bounds["xrange"]["lb"]) / mdp_env_params[
        "grid_width"
    ]
    mdp_env_params["cell_size"]["y"] = (world_bounds["yrange"]["ub"] - world_bounds["yrange"]["lb"]) / mdp_env_params[
        "grid_depth"
    ]
    mdp_env_params["cell_size"]["z"] = (world_bounds["zrange"]["ub"] - world_bounds["zrange"]["lb"]) / mdp_env_params[
        "grid_height"
    ]

    num_objs = 4
    obj_positions = np.array([[0] * 3] * num_objs, dtype="f")
    obj_quats = np.array([[0] * 4] * num_objs, dtype="f")
    obj_positions[0][0] = 0.382  # custom left otp
    obj_positions[0][1] = -0.122
    obj_positions[0][2] = 0.042
    obj_quats[0][0] = 0.991
    obj_quats[0][1] = 0.086
    obj_quats[0][2] = 0.080
    obj_quats[0][3] = -0.059

    obj_positions[1][0] = 0.344  # custom left otp
    obj_positions[1][1] = -0.525
    obj_positions[1][2] = 0.089
    obj_quats[1][0] = 0.990
    obj_quats[1][1] = 0.100
    obj_quats[1][2] = 0.082
    obj_quats[1][3] = -0.051

    obj_positions[2][0] = -0.391  # custom left otp
    obj_positions[2][1] = -0.574
    obj_positions[2][2] = 0.072
    obj_quats[2][0] = 0.644
    obj_quats[2][1] = -0.358
    obj_quats[2][2] = -0.206
    obj_quats[2][3] = 0.644

    obj_positions[3][0] = -0.330  # custom left otp
    obj_positions[3][1] = -0.570
    obj_positions[3][2] = 0.502
    obj_quats[3][0] = -0.644
    obj_quats[3][1] = -0.278
    obj_quats[3][2] = 0.673
    obj_quats[3][3] = -0.236
    env_params = dict()
    env_params["all_mdp_env_params"] = mdp_env_params
    env_params["mdp_list"] = mdp_list
    env_params["spatial_window_half_length"] = 2
    env_params["goal_positions"] = obj_positions
    env_params["goal_quats"] = obj_quats
    disamb_algo = DiscreteMIDisambAlgo3D(env_params, "andreddwtest")

    cell_size = mdp_env_params["cell_size"]
    query_state = (4, 3, 3, 5)
    start_coord = (4, 3, 3, 2)
    # belief = [0.47, 0.47, 0.03, 0.03]

    belief = [0.01, 0.01, 0.48, 0.48]
    print("0")
    print(" ")
    simulate_trajectories(start_coord, 0, mdp_list)
    print("1")
    print(" ")
    simulate_trajectories(start_coord, 1, mdp_list)
    print("2")
    print(" ")
    simulate_trajectories(start_coord, 2, mdp_list)
    print("3")
    print(" ")
    simulate_trajectories(start_coord, 3, mdp_list)
    robot_position = convert_discrete_state_to_continuous_position(query_state, cell_size, world_bounds)
    max_disamb_state = disamb_algo.get_local_disamb_state(belief, query_state, robot_position)
    print(max_disamb_state)

    print("DISMAB 0")
    print(" ")
    simulate_trajectories(max_disamb_state, 0, mdp_list)
    print("DISMAB1")
    print(" ")
    simulate_trajectories(max_disamb_state, 1, mdp_list)
    print("DISMAB2")
    print(" ")
    simulate_trajectories(max_disamb_state, 2, mdp_list)
    print("DISMAB3")
    print(" ")
    simulate_trajectories(max_disamb_state, 3, mdp_list)
