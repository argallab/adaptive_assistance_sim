#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import collections
import os
import sys
import rospkg

sys.path.append(os.path.join(rospkg.RosPack().get_path("simulators"), "scripts"))
from generate_adaptive_assistance_trials import create_random_obstacles, create_random_goals, create_random_start_state
from mdp.mdp_discrete_SE2_NH_gridworld import MDPDiscrete_SE2_NH_GridWorld
from adaptive_assistance_sim_utilss import *
import matplotlib.pyplot as plt
import copy

NUM_GOALS = 3
OCCUPANCY_LEVEL = 0.13
MAX_PATCHES = 4

GRID_WIDTH = 20
GRID_HEIGHT = 30


def visualize_grid(mdp):
    mdp_params = mdp.get_env_params()
    grid = np.zeros((mdp_params["grid_width"], mdp_params["grid_height"]))
    for obs in mdp_params["mdp_obstacles"]:
        grid[obs[0], obs[1]] = 1

    gs = mdp_params["mdp_goal_state"]
    grid[gs[0], gs[1]] = 0.5
    grid = grid.T
    grid = np.flipud(grid)
    plt.imshow(grid)
    plt.colorbar()
    plt.show()


def visualize_V_and_policy(mdp):
    mdp_params = mdp.get_env_params()
    V = np.array(mdp.get_value_function()).reshape(
        (mdp_params["grid_width"], mdp_params["grid_height"], mdp_params["num_discrete_orientations"])
    )
    fig, ax = plt.subplots(2, mdp_params["num_discrete_orientations"])
    for i in range(mdp_params["num_discrete_orientations"]):
        Va = np.flipud(V[:, :, i].T)
        vmin = np.percentile(Va, 1)
        vmax = np.percentile(Va, 99)

        im = ax[0, i].imshow(Va, vmin=vmin, vmax=vmax)
        cbar = ax[0, i].figure.colorbar(im, ax=ax[0, i])
        cbar.ax.set_ylabel("V", rotation=-90, va="bottom")

    _, policy_dict = mdp.get_optimal_policy_for_mdp()
    grids = np.zeros((mdp_params["num_discrete_orientations"], mdp_params["grid_width"], mdp_params["grid_height"]))
    for s in policy_dict.keys():
        grids[s[-1] - 1, s[0], s[1]] = policy_dict[
            s
        ]  # first index is the mode. modes are 1 and 2. hence the -1 to account for 0-indexing

    for i in range(mdp_params["num_discrete_orientations"]):
        grid = grids[i, :, :]
        grid = grid.T
        grid = np.flipud(grid)
        im = ax[1, i].imshow(grid)
        ax[1, i].set_title("Learned Policy Map")
        cbar = ax[1, i].figure.colorbar(im, ax=ax[1, i])
        cbar.ax.set_ylabel("cc-c-f-b", rotation=90, va="bottom")

    # fig.tight_layout()
    plt.show()


def visualize_trajectory(sas_trajectory, mdp_env_params):
    pass


def create_mdp_env_param_dict():
    mdp_env_params = collections.OrderedDict()
    mdp_env_params["rl_algo_type"] = RlAlgoType.ValueIteration
    mdp_env_params["gamma"] = 0.9
    mdp_env_params["grid_width"] = GRID_WIDTH
    mdp_env_params["grid_height"] = GRID_HEIGHT
    mdp_env_params["num_discrete_orientations"] = 8
    mdp_env_params["original_mdp_obstacles"] = []  # create_random_obstacles(  width=mdp_env_params['grid_width'],
    # height=mdp_env_params['grid_height'],
    # occupancy_measure=OCCUPANCY_LEVEL,
    # num_obstacle_patches=np.random.randint(MAX_PATCHES) + 1)

    g_list = create_random_goals(
        width=mdp_env_params["grid_width"],
        height=mdp_env_params["grid_height"],
        num_goals=NUM_GOALS,
        obstacle_list=mdp_env_params["original_mdp_obstacles"],
    )  # make the list a tuple

    # g = list(g)
    for i, g in enumerate(g_list):
        g = list(g)
        g.append(np.random.randint(mdp_env_params["num_discrete_orientations"]))
        g_list[i] = tuple(g)
    # g.append(np.random.randint(mdp_env_params['num_discrete_orientations']))
    # mdp_env_params['mdp_goal_state'] = (1, 0, 0)

    mdp_env_params["all_goals"] = g_list  # list of tuples
    # print(mdp_env_params['mdp_goal_state']) #(3d goal state)
    mdp_env_params["obstacle_penalty"] = -1000
    mdp_env_params["goal_reward"] = 100
    mdp_env_params["step_penalty"] = -1
    mdp_env_params["rand_direction_factor"] = 0.0

    return mdp_env_params


def simulate_human_SE2_NH_mdp():
    mdp_env_params = create_mdp_env_param_dict()
    mdp_list = []
    width = mdp_env_params["grid_width"]
    height = mdp_env_params["grid_height"]

    def get_all_neighbors(cell):
        def check_bounds(state):
            state[0] = max(0, min(state[0], width - 1))
            state[1] = max(0, min(state[1], height - 1))
            return state

        top_neighbor = tuple(check_bounds(np.array(cell) + (0, 1)))
        bottom_neighbor = tuple(check_bounds(np.array(cell) + (0, -1)))
        left_neighbor = tuple(check_bounds(np.array(cell) + (-1, 0)))
        right_neighbor = tuple(check_bounds(np.array(cell) + (1, 0)))

        top_right_neighbor = tuple(check_bounds(np.array(cell) + (1, 1)))
        top_left_neighbor = tuple(check_bounds(np.array(cell) + (-1, 1)))
        bottom_left_neighbor = tuple(check_bounds(np.array(cell) + (-1, -1)))
        bottom_right_neighbor = tuple(check_bounds(np.array(cell) + (1, -1)))

        all_neighbors = [
            top_neighbor,
            bottom_neighbor,
            left_neighbor,
            right_neighbor,
            top_right_neighbor,
            top_left_neighbor,
            bottom_left_neighbor,
            bottom_right_neighbor,
        ]
        # return all_neighbors
        return list(set(all_neighbors))

    # import IPython; IPython.embed(banner1='check goals')
    for i, g in enumerate(mdp_env_params["all_goals"]):
        mdp_env_params["mdp_goal_state"] = g
        goals_that_are_obs = [(g_obs[0], g_obs[1]) for g_obs in mdp_env_params["all_goals"] if g_obs != g]
        # get some neighbors of these goals
        all_neighbors_of_g_obs = []
        for g_obs in goals_that_are_obs:
            neighbors_of_g_obs = get_all_neighbors(g_obs)
            all_neighbors_of_g_obs.append(neighbors_of_g_obs)

        mdp_env_params["mdp_obstacles"] = copy.deepcopy(mdp_env_params["original_mdp_obstacles"])
        mdp_env_params["mdp_obstacles"].extend(goals_that_are_obs)
        for neighbors_of_g_obs in all_neighbors_of_g_obs:
            mdp_env_params["mdp_obstacles"].extend(neighbors_of_g_obs)

        discrete_SE2_NH_modes_mdp = MDPDiscrete_SE2_NH_GridWorld(copy.deepcopy(mdp_env_params))
        visualize_grid(discrete_SE2_NH_modes_mdp)
        visualize_V_and_policy(discrete_SE2_NH_modes_mdp)
        mdp_list.append(discrete_SE2_NH_modes_mdp)

    # discrete_SE2_NH_mdp = MDPDiscrete_SE2_NH_GridWorld(mdp_env_params)

    # visualize_grid(discrete_SE2_NH_mdp)
    # visualize_V_and_policy(discrete_SE2_NH_mdp)


if __name__ == "__main__":
    simulate_human_SE2_NH_mdp()
