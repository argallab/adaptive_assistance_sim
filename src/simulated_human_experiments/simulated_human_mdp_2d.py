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

GRID_WIDTH = 15
GRID_HEIGHT = 30
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
    # mdp_env_params["all_goals"] = create_random_goals(
    #     width=mdp_env_params["grid_width"],
    #     height=mdp_env_params["grid_height"],
    #     num_goals=NUM_GOALS,
    #     obstacle_list=mdp_env_params["original_mdp_obstacles"],
    # )
    mdp_env_params["all_goals"] = [(3, 18), (8, 5), (14, 6)]
    # mdp_env_params["all_goals"] = [(5, 1), (5, 8), (7, 8)]
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


def visualize_metrics_map(metric_dict, mdp, title="default"):

    mdp_params = mdp.get_env_params()
    grids = np.zeros((mdp_params["num_modes"], mdp_params["grid_width"], mdp_params["grid_height"]))
    fig, ax = plt.subplots(1, mdp_params["num_modes"])
    for k in metric_dict.keys():
        grids[k[-1] - 1, k[0], k[1]] = metric_dict[k]
    for i in range(mdp_params["num_modes"]):
        grid = grids[i, :, :]
        grid = grid.T
        grid = np.flipud(grid)
        im = ax[i].imshow(grid)
        ax[i].set_title("{} Map for mode {}".format(title, i + 1))
        cbar = ax[i].figure.colorbar(im, ax=ax[i])
    fig.tight_layout()
    plt.show()


def visualize_V_and_policy(mdp):
    mdp_params = mdp.get_env_params()
    V = np.array(mdp.get_value_function()).reshape(
        (mdp_params["grid_width"], mdp_params["grid_height"], mdp_params["num_modes"])
    )
    fig, ax = plt.subplots(2, mdp_params["num_modes"])
    for i in range(mdp_params["num_modes"]):
        Va = np.flipud(V[:, :, i].T)
        vmin = np.percentile(Va, 1)
        vmax = np.percentile(Va, 99)

        im = ax[0, i].imshow(Va, vmin=vmin, vmax=vmax)
        cbar = ax[0, i].figure.colorbar(im, ax=ax[0, i])
        cbar.ax.set_ylabel("V", rotation=-90, va="bottom")

    _, policy_dict = mdp.get_optimal_policy_for_mdp()
    grids = np.zeros((mdp_params["num_modes"], mdp_params["grid_width"], mdp_params["grid_height"]))
    for s in policy_dict.keys():
        # first index is the mode. modes are 1 and 2. hence the -1 to account for 0-indexing
        grids[s[-1] - 1, s[0], s[1]] = policy_dict[s]

    for i in range(mdp_params["num_modes"]):
        grid = grids[i, :, :]
        grid = grid.T
        grid = np.flipud(grid)
        im = ax[1, i].imshow(grid)
        ax[1, i].set_title("Learned Policy Map")
        cbar = ax[1, i].figure.colorbar(im, ax=ax[1, i])
        cbar.ax.set_ylabel("mp, mn, mode_r, mode_l", rotation=90, va="bottom")

    fig.tight_layout()
    plt.show()


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

    # for mdp in mdp_list:
    #     visualize_V_and_policy(mdp)
    disamb_algo = DiscreteMIDisambAlgo2D(env_params, "adnadnak")
    prior = [1 / NUM_GOALS] * NUM_GOALS
    states_for_disamb_computation = mdp_list[0].get_all_state_coords_with_grid_locs_diff_from_goals_and_obs()
    continuous_positions_of_local_spatial_window = [
        convert_discrete_state_to_continuous_position(s, mdp_env_params["cell_size"], world_bounds)
        for s in states_for_disamb_computation
    ]
    disamb_algo._compute_mi(prior, states_for_disamb_computation, continuous_positions_of_local_spatial_window)

    visualize_metrics_map(disamb_algo.avg_mi_for_valid_states, mdp_list[0], title="MI")
    visualize_metrics_map(disamb_algo.dist_of_vs_from_weighted_mean_of_goals, mdp_list[0], title="WEIGHTED DIST")
    visualize_metrics_map(disamb_algo.avg_total_reward_for_valid_states, mdp_list[0], title="REWARD")
