#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import collections
import os
from scipy import special
import sys
import copy
from scipy.stats import entropy
import itertools

sys.path.append(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "mdp", "src"))
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "disamb_algo", "src"))

from mdp.mdp_discrete_2d_gridworld_with_modes import MDPDiscrete2DGridWorldWithModes
from adaptive_assistance_sim_utils import *
from mdp.mdp_utils import *
import matplotlib.pyplot as plt
from disamb_algo.discrete_mi_disamb_algo_2d import DiscreteMIDisambAlgo2D

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


NUM_GOALS = 3
OCCUPANCY_LEVEL = 0.0
MAX_PATCHES = 4

GRID_WIDTH = 10
GRID_HEIGHT = 10
ENTROPY_THRESHOLD = 0.6
SPATIAL_WINDOW_HALF_LENGTH = 3

# DICTIONARIES
DIM_TO_MODE_INDEX_XYZ = {"x": 0, "y": 1, "z": 2, "gr": 3}
MODE_INDEX_TO_DIM_XYZ = {v: k for k, v in DIM_TO_MODE_INDEX.items()}


def create_mdp_env_param_dict():
    mdp_env_params = collections.OrderedDict()
    mdp_env_params["rl_algo_type"] = RlAlgoType.ValueIteration
    mdp_env_params["gamma"] = 0.96
    mdp_env_params["grid_width"] = GRID_WIDTH
    mdp_env_params["grid_height"] = GRID_HEIGHT

    mdp_env_params["robot_type"] = CartesianRobotType.R2
    mdp_env_params["mode_set_type"] = ModeSetType.OneD
    num_patches = 2
    # mdp_env_params["original_mdp_obstacles"] = create_random_obstacles(
    #     width=mdp_env_params["grid_width"],
    #     height=mdp_env_params["grid_height"],
    #     occupancy_measure=OCCUPANCY_LEVEL,
    #     num_obstacle_patches=num_patches,
    # )

    mdp_env_params["original_mdp_obstacles"] = []
    # make the list a tuple
    mdp_env_params["all_goals"] = create_random_goals(
        width=mdp_env_params["grid_width"],
        height=mdp_env_params["grid_height"],
        num_goals=NUM_GOALS,
        obstacle_list=mdp_env_params["original_mdp_obstacles"],
    )
    # mdp_env_params["all_goals"] = [(3, 18), (8, 5), (14, 6)]
    # mdp_env_params['all_goals'] = [(3, 15), (12, 15)]
    # print(mdp_env_params['mdp_goal_state']) #(2d goal state)
    # mdp_env_params['all_goals'] = [(0,0), (0,GRID_HEIGHT-1), (GRID_WIDTH-1, GRID_HEIGHT-1)]
    mdp_env_params["obstacle_penalty"] = -100
    mdp_env_params["goal_reward"] = 100
    mdp_env_params["step_penalty"] = -10
    mdp_env_params["rand_direction_factor"] = 0.0
    mdp_env_params["sparsity_factor"] = 0.0
    mdp_env_params["mdp_obstacles"] = []

    return mdp_env_params


def create_mdp_list(mdp_env_params):
    mdp_list = []
    for i, g in enumerate(mdp_env_params["all_goals"]):
        mdp_env_params["mdp_goal_state"] = g
        goals_that_are_obs = [(g_obs[0], g_obs[1]) for g_obs in mdp_env_params["all_goals"] if g_obs != g]
        mdp_env_params["mdp_obstacles"] = copy.deepcopy(mdp_env_params["original_mdp_obstacles"])
        mdp_env_params["mdp_obstacles"].extend(goals_that_are_obs)
        discrete_2d_modes_mdp = MDPDiscrete2DGridWorldWithModes(copy.deepcopy(mdp_env_params))
        print("GOAL", g)
        # visualize_grid(discrete_2d_modes_mdp)
        # visualize_V_and_policy(discrete_2d_modes_mdp)
        mdp_list.append(discrete_2d_modes_mdp)

    return mdp_list


def convert_discrete_state_to_continuous_position(discrete_state, cell_size, world_bounds):
    x_coord = discrete_state[0]
    y_coord = discrete_state[1]

    position = [
        x_coord * cell_size["x"] + cell_size["x"] / 2.0 + world_bounds["xrange"]["lb"],
        y_coord * cell_size["y"] + cell_size["y"] / 2.0 + world_bounds["yrange"]["lb"],
    ]

    return position


if __name__ == "__main__":
    mdp_env_params = create_mdp_env_param_dict()

    mdp_list = create_mdp_list(mdp_env_params)

    world_bounds = collections.OrderedDict()
    world_bounds["xrange"] = collections.OrderedDict()
    world_bounds["yrange"] = collections.OrderedDict()
    # bottom left corner in continuous space
    world_bounds["xrange"]["lb"] = 0.05 * VIEWPORT_W / SCALE
    world_bounds["yrange"]["lb"] = 0.05 * VIEWPORT_H / SCALE
    world_bounds["xrange"]["ub"] = 0.75 * VIEWPORT_W / SCALE
    world_bounds["yrange"]["ub"] = 0.9 * VIEWPORT_H / SCALE

    mdp_env_params["cell_size"] = collections.OrderedDict()
    mdp_env_params["cell_size"]["x"] = (world_bounds["xrange"]["ub"] - world_bounds["xrange"]["lb"]) / mdp_env_params[
        "grid_width"
    ]
    mdp_env_params["cell_size"]["y"] = (world_bounds["yrange"]["ub"] - world_bounds["yrange"]["lb"]) / mdp_env_params[
        "grid_height"
    ]

    env_params = collections.OrderedDict()
    env_params["all_mdp_env_params"] = mdp_env_params
    env_params["robot_type"] = CartesianRobotType.R2
    env_params["mode_set_type"] = ModeSetType.OneD
    env_params["spatial_window_half_length"] = 3

    def _generate_continuous_goal_poses(discrete_goal_list, cell_size, world_bounds):
        goal_poses = []
        for dg in discrete_goal_list:
            goal_pose = [0, 0]
            goal_pose[0] = (dg[0] * cell_size["x"]) + cell_size["x"] / 2.0 + world_bounds["xrange"]["lb"]
            goal_pose[1] = (dg[1] * cell_size["y"]) + cell_size["y"] / 2.0 + world_bounds["yrange"]["lb"]
            goal_poses.append(goal_pose)

        return goal_poses

    env_params["goal_poses"] = _generate_continuous_goal_poses(
        mdp_env_params["all_goals"], mdp_env_params["cell_size"], world_bounds
    )
    env_params["mdp_list"] = mdp_list

    disamb_algo = DiscreteMIDisambAlgo2D(env_params, "test")

    states_for_disamb_computation = mdp_list[0].get_all_state_coords_with_grid_locs_diff_from_goals_and_obs()
    continuous_positions_of_local_spatial_window = [
        convert_discrete_state_to_continuous_position(s, mdp_env_params["cell_size"], world_bounds)
        for s in states_for_disamb_computation
    ]

    import IPython

    IPython.embed(banner1="check params")
