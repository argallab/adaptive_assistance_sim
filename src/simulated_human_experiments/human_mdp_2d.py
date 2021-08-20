#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import collections
import os
import sys
import rospkg

sys.path.append(os.path.join(rospkg.RosPack().get_path("simulators"), "scripts"))
from generate_adaptive_assistance_trials import create_random_obstacles, create_random_goals, create_random_start_state
from mdp.mdp_discrete_2d_gridworld import MDPDiscrete2DGridWorld
from adaptive_assistance_sim_utils import *
import matplotlib.pyplot as plt

NUM_GOALS = 1
OCCUPANCY_LEVEL = 0.13
MAX_PATCHES = 4

GRID_WIDTH = 15
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
    V = np.array(mdp.get_value_function()).reshape((GRID_WIDTH, GRID_HEIGHT)).T  # height, width
    V = np.flipud(V)
    vmin = np.percentile(V, 1)
    vmax = np.percentile(V, 99)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    im1 = ax1.imshow(V, vmin=vmin, vmax=vmax)
    ax1.set_title("Value Function Map")
    cbar = ax1.figure.colorbar(im1, ax=ax1)
    cbar.ax.set_ylabel("V", rotation=-90, va="bottom")

    _, policy_dict = mdp.get_optimal_policy_for_mdp()
    grid = np.zeros((mdp_params["grid_width"], mdp_params["grid_height"]))
    for s in policy_dict.keys():
        grid[s[0], s[1]] = policy_dict[s]

    grid = grid.T
    grid = np.flipud(grid)
    im2 = ax2.imshow(grid)
    ax2.set_title("Learned Policy Map")
    cbar = ax2.figure.colorbar(im2, ax=ax2)
    cbar.ax.set_ylabel("u-d-l-r", rotation=90, va="bottom")
    fig.tight_layout()

    plt.show()


def visualize_trajectory(sas_traj, mdp):
    # mark obstacles
    mdp_params = mdp.get_env_params()
    fig = plt.figure()
    obstacle_list = mdp_params["mdp_obstacles"]
    for obs in obstacle_list:
        plt.plot(obs[0], obs[1], "rx", linewidth=2.0)

    for t, sas_t in enumerate(sas_traj):
        dx = sas_t[2][0] - sas_t[0][0] + 0.00001
        dy = sas_t[2][1] - sas_t[0][1] + 0.00001
        plt.arrow(
            sas_t[0][0],
            sas_t[0][1],
            dx,
            dy,
            fc="k",
            ec="k",
            head_width=0.1,
            head_length=0.1,
            length_includes_head=True,
        )

    plt.scatter(sas_traj[0][0][0], sas_traj[0][0][1], s=100, c="b")
    plt.scatter(sas_traj[-1][-1][0], sas_traj[-1][-1][1], s=100, c="r")
    plt.axes().set_aspect("equal")
    plt.xlim(0, mdp_params["grid_width"])
    plt.ylim(0, mdp_params["grid_height"])

    # fig.tight_layout()
    plt.show()


def create_mdp_env_param_dict():
    # (0,0) bottom left
    mdp_env_params = collections.OrderedDict()
    mdp_env_params["rl_algo_type"] = RlAlgoType.ValueIteration
    mdp_env_params["gamma"] = 0.96
    mdp_env_params["grid_width"] = GRID_WIDTH
    mdp_env_params["grid_height"] = GRID_HEIGHT
    mdp_env_params["mdp_obstacles"] = create_random_obstacles(
        width=mdp_env_params["grid_width"],
        height=mdp_env_params["grid_height"],
        occupancy_measure=OCCUPANCY_LEVEL,
        num_obstacle_patches=np.random.randint(MAX_PATCHES) + 1,
    )

    mdp_env_params["mdp_goal_state"] = create_random_goals(
        width=mdp_env_params["grid_width"],
        height=mdp_env_params["grid_height"],
        num_goals=NUM_GOALS,
        obstacle_list=mdp_env_params["mdp_obstacles"],
    )[
        0
    ]  # make the list a tuple
    print(mdp_env_params["mdp_goal_state"])
    mdp_env_params["obstacle_penalty"] = -100
    mdp_env_params["goal_reward"] = 100
    mdp_env_params["step_penalty"] = -10

    return mdp_env_params


def simulate_human_2d_mdp():
    mdp_env_params = create_mdp_env_param_dict()
    discrete_2D_MDP = MDPDiscrete2DGridWorld(mdp_env_params)

    visualize_grid(discrete_2D_MDP)
    visualize_V_and_policy(discrete_2D_MDP)

    # trajectory generation
    random_state = discrete_2D_MDP.get_random_valid_state()
    opt_traj_from_random_state = discrete_2D_MDP.get_optimal_trajectory_from_state(random_state)
    visualize_trajectory(opt_traj_from_random_state, discrete_2D_MDP)

    # random_state = discrete_2D_MDP.get_random_valid_state()
    # rand_traj_from_random_state = discrete_2D_MDP.get_random_trajectory_from_state(random_state, horizon=20)
    # visualize_trajectory(rand_traj_from_random_state, discrete_2D_MDP)


if __name__ == "__main__":
    simulate_human_2d_mdp()
