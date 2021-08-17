#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import collections
import os
from scipy import special
import sys
import rospkg
import copy

sys.path.append(os.path.join(rospkg.RosPack().get_path("simulators"), "scripts"))
from generate_adaptive_assistance_trials import create_random_obstacles, create_random_goals, create_random_start_state
from mdp.mdp_discrete_2d_gridworld_with_modes import MDPDiscrete2DGridWorldWithModes
from envs.utils import *
import matplotlib.pyplot as plt
from intent_inference.intent_inference_engine import IntentInference

NUM_GOALS = 3
OCCUPANCY_LEVEL = 0.0
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


def visualize_V_expectation(mdp_list, prior):
    assert len(mdp_list) == len(prior)
    assert len(mdp_list) > 0
    mdp = mdp_list[0]
    mdp_params = mdp.get_env_params()
    V = prior[0] * np.array(mdp.get_value_function()).reshape(
        (mdp_params["grid_width"], mdp_params["grid_height"], mdp_params["num_modes"])
    )
    for i, mdp in enumerate(mdp_list[1:]):
        mdp_params = mdp.get_env_params()
        V += prior[i + 1] * np.array(mdp.get_value_function()).reshape(
            (mdp_params["grid_width"], mdp_params["grid_height"], mdp_params["num_modes"])
        )

    fig, ax = plt.subplots(1, mdp_params["num_modes"])
    for i in range(mdp_params["num_modes"]):
        Va = np.flipud(V[:, :, i].T)
        vmin = np.percentile(Va, 1)
        vmax = np.percentile(Va, 99)

        im = ax[i].imshow(Va, vmin=vmin, vmax=vmax)
        cbar = ax[i].figure.colorbar(im, ax=ax[i])
        cbar.ax.set_ylabel("V", rotation=-90, va="bottom")

    fig.tight_layout()
    plt.show()

    return V


def visualize_trajectory(sas_trajectory_list, mdp):
    mdp_params = mdp.get_env_params()
    fig = plt.figure()
    obstacle_list = mdp_params["mdp_obstacles"]
    for obs in obstacle_list:
        plt.plot(obs[0], obs[1], "rx", linewidth=2.0)
    for sas_traj in sas_trajectory_list:
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


def visualize_metrics_map(metric_dict, mdp, title="default"):
    mdp_params = mdp.get_env_params()
    grids = np.zeros((mdp_params["num_modes"], mdp_params["grid_width"], mdp_params["grid_height"]))
    fig, ax = plt.subplots(1, mdp_params["num_modes"])
    for k in metric_dict.keys():
        grids[k[-1] - 1, k[0], k[1]] = metric_dict[k]
    # for i in range(mdp_params['num_modes']):
    #     grid = grids[i, :, :]
    #     grid = grid.T
    #     grid = np.flipud(grid)
    #     im = ax[0, i].imshow(grid)
    #     ax[0, i].set_title('{} Map for mode {}'.format(title, i+1))
    #     cbar = ax[0, i].figure.colorbar(im, ax=ax[0,i])
    #     # cbar.ax.set_ylabel("", rotation=90, va="bottom")
    for i in range(mdp_params["num_modes"]):
        grid = grids[i, :, :]
        grid = grid.T
        grid = np.flipud(grid)
        im = ax[i].imshow(grid)
        ax[i].set_title("{} Map for mode {}".format(title, i + 1))
        cbar = ax[i].figure.colorbar(im, ax=ax[i])
    fig.tight_layout()
    plt.show()


def create_mdp_env_param_dict():
    mdp_env_params = collections.OrderedDict()
    mdp_env_params["rl_algo_type"] = RlAlgoType.ValueIteration
    mdp_env_params["gamma"] = 0.96
    mdp_env_params["grid_width"] = GRID_WIDTH
    mdp_env_params["grid_height"] = GRID_HEIGHT
    mdp_env_params["robot_type"] = CartesianRobotType.R2
    mdp_env_params["mode_set_type"] = ModeSetType.OneD
    num_patches = 2
    # mdp_env_params['original_mdp_obstacles'] = create_random_obstacles(width=mdp_env_params['grid_width'],
    #                                                             height=mdp_env_params['grid_height'],
    #                                                             occupancy_measure=OCCUPANCY_LEVEL,
    #                                                             num_obstacle_patches=num_patches)

    mdp_env_params["original_mdp_obstacles"] = []
    # mdp_env_params['all_goals'] = create_random_goals( width=mdp_env_params['grid_width'],
    #                                                         height=mdp_env_params['grid_height'],
    #                                                         num_goals=NUM_GOALS,
    #                                                         obstacle_list=mdp_env_params['original_mdp_obstacles']) #make the list a tuple
    # import IPython; IPython.embed(banner1='chck')
    mdp_env_params["all_goals"] = [(3, 18), (8, 5), (14, 6)]
    # mdp_env_params['all_goals'] = [(3, 15), (12, 15)]
    # print(mdp_env_params['mdp_goal_state']) #(2d goal state)
    # mdp_env_params['all_goals'] = [(0,0), (0,GRID_HEIGHT-1), (GRID_WIDTH-1, GRID_HEIGHT-1)]
    mdp_env_params["obstacle_penalty"] = -100
    mdp_env_params["goal_reward"] = 100
    mdp_env_params["step_penalty"] = -10
    mdp_env_params["rand_direction_factor"] = 0.4
    mdp_env_params["sparsity_factor"] = 0.0
    mdp_env_params["mdp_obstacles"] = []

    return mdp_env_params


def simulate_turn_taking_and_inference():

    mdp_env_params = create_mdp_env_param_dict()
    mdp_list = []
    for i, g in enumerate(mdp_env_params["all_goals"]):
        mdp_env_params["mdp_goal_state"] = g
        goals_that_are_obs = [(g_obs[0], g_obs[1]) for g_obs in mdp_env_params["all_goals"] if g_obs != g]
        mdp_env_params["mdp_obstacles"] = copy.deepcopy(mdp_env_params["original_mdp_obstacles"])
        mdp_env_params["mdp_obstacles"].extend(goals_that_are_obs)
        discrete_2d_modes_mdp = MDPDiscrete2DGridWorldWithModes(copy.deepcopy(mdp_env_params))

        visualize_grid(discrete_2d_modes_mdp)
        visualize_V_and_policy(discrete_2d_modes_mdp)
        mdp_list.append(discrete_2d_modes_mdp)

    # TODO
    ii_engine = IntentInference(mdp_list, use_state_information=False)
    # initialize inference engine with all goals.

    current_state = mdp_list[0].get_random_valid_state()  # pick a random state to start with.
    horizon = 100
    num_steps_per_turn = 5  # for human
    steps_per_turn_counter = 0
    p_vec = [1.0 / NUM_GOALS] * NUM_GOALS  # uninformed prior

    g = np.random.choice(NUM_GOALS, p=p_vec)
    print("ALL GOALS ", mdp_env_params["all_goals"])
    print("RANDOM GOAL CHOICE ", g)
    print("Starting belief", ii_engine.get_current_p_g_given_a_s())
    for t in range(horizon):
        action = mdp_list[g].get_optimal_action(current_state, return_optimal=False)  # get noisily optimal action

        # package all necessary information for inference enginer
        inference_info_dict = {}
        inference_info_dict["action"] = action
        inference_info_dict["state"] = current_state
        ii_engine.perform_inference(inference_info_dict)  # ii_engine internally maintains the belif
        print("Current state, action, belief", current_state, action, ii_engine.get_current_p_g_given_a_s())
        next_state = mdp_list[g].get_next_state_from_state_action(current_state, action)
        if mdp_list[g].check_if_state_coord_is_goal_state(next_state):
            print("Reached goal")
            break
        steps_per_turn_counter += 1
        if steps_per_turn_counter >= num_steps_per_turn:
            steps_per_turn_counter = 0
            # time for autonomous agent to analyze the belief and make decision regarding whether to perform intent disambiguation of drive the robot to the inferred goal.
            # dsicretize the belief for more robustness for MC computation.
            import IPython

            IPython.embed(banner1="perform intentt disambiguation or shared control assistance. ")

        current_state = next_state


def create_mdp_list(mdp_env_params):
    mdp_list = []
    for i, g in enumerate(mdp_env_params["all_goals"]):
        mdp_env_params["mdp_goal_state"] = g
        goals_that_are_obs = [(g_obs[0], g_obs[1]) for g_obs in mdp_env_params["all_goals"] if g_obs != g]
        mdp_env_params["mdp_obstacles"] = copy.deepcopy(mdp_env_params["original_mdp_obstacles"])
        mdp_env_params["mdp_obstacles"].extend(goals_that_are_obs)
        discrete_2d_modes_mdp = MDPDiscrete2DGridWorldWithModes(copy.deepcopy(mdp_env_params))
        print("GOAL", g)
        visualize_grid(discrete_2d_modes_mdp)
        visualize_V_and_policy(discrete_2d_modes_mdp)
        mdp_list.append(discrete_2d_modes_mdp)

    return mdp_list


def check_vis_traj():
    mdp_env_params = create_mdp_env_param_dict()
    mdp_list = create_mdp_list(mdp_env_params)
    g = 0
    num_trajectories = 100
    traj_list = []
    for i in range(num_trajectories):
        random_state_state = mdp_list[g].get_random_valid_state()
        opt_traj_from_state = mdp_list[g].get_optimal_trajectory_from_state(
            random_state_state, horizon=100, return_optimal=True
        )
        traj_list.append(opt_traj_from_state)

    visualize_trajectory(traj_list, mdp_list[g])
    g = 1
    traj_list_1 = []
    for i in range(num_trajectories):
        random_state_state = mdp_list[g].get_random_valid_state()
        opt_traj_from_state = mdp_list[g].get_optimal_trajectory_from_state(
            random_state_state, horizon=100, return_optimal=True
        )
        traj_list_1.append(opt_traj_from_state)

    visualize_trajectory(traj_list_1, mdp_list[g])


def compute_mi(mdp_list, mdp_env_params, prior):
    all_valid_states = mdp_list[0].get_all_state_coords()
    avg_mi_for_valid_states = collections.OrderedDict()
    avg_dist_for_valid_states_from_goals = collections.OrderedDict()
    avg_total_reward_for_valid_states = collections.OrderedDict()
    num_trajectories = 100
    assert len(prior) == NUM_GOALS

    kl_coeff = 1.0
    dist_coeff = 0.0
    for vs in all_valid_states:
        print("Computing MI for ", vs)
        traj_list = collections.defaultdict(list)
        for i in range(num_trajectories):
            sampled_goal_index = np.random.choice(NUM_GOALS, p=prior)  # sample a goal according to current belief
            mdp_for_sampled_goal = mdp_list[sampled_goal_index]
            # one step is all you need
            opt_traj_from_state = mdp_for_sampled_goal.get_optimal_trajectory_from_state(
                vs, horizon=1, return_optimal=True
            )
            traj_list[sampled_goal_index].append(opt_traj_from_state)

        action_to_action_id_map = mdp_for_sampled_goal.get_action_to_action_id_map()

        import IPython

        IPython.embed(banner1="check map")
        # actions are in task-space
        p_a_g_s0 = collections.defaultdict(list)  # p(a | g, s0)
        for g in traj_list.keys():
            for traj_g in traj_list[g]:
                a0 = traj_g[0][1]
                p_a_g_s0[g].append(action_to_action_id_map[a0])

        p_a_s0 = []  # p(a | s)
        for g in p_a_g_s0.keys():
            p_a_s0.extend(p_a_g_s0[g])

        action_ids = mdp_list[0].get_action_to_action_id_map().values()

        pa_s0_hist = collections.Counter(p_a_s0)
        for action in action_ids:
            if action not in pa_s0_hist.keys():
                pa_s0_hist[action] = 0

        pa_s = np.array(pa_s0_hist.values(), dtype=np.float32)
        pa_s = pa_s / np.sum(pa_s)  # for current valid state. the action distributoin
        kl_list = []

        for g in p_a_g_s0.keys():
            pa_gs_hist = collections.Counter(p_a_g_s0[g])
            for action in action_ids:  # make sure all actions have a count.
                if action not in pa_gs_hist.keys():
                    pa_gs_hist[action] = 0

            assert len(pa_gs_hist) == len(pa_s)

            pa_sg = np.array(pa_gs_hist.values(), dtype=np.float32)
            pa_sg = pa_sg / np.sum(pa_sg)
            kl = np.sum(special.rel_entr(pa_sg, pa_s))  # log e #compute the d_kl(p(a|s,g) || p(a|s))
            kl_list.append(kl)

        # normalized to grid dimensions
        avg_dist_of_vs_from_goals = np.mean(
            np.linalg.norm(
                (np.array(mdp_env_params["all_goals"]) - np.array(vs[:2]))
                / np.array([GRID_WIDTH, GRID_HEIGHT], dtype=np.float32),
                axis=1,
            )
        )
        # the state that is the "centroid of the triangle joining the goals will be the state with minimum avg distance. but needn't be the state with maximum MI"
        avg_mi_for_valid_states[vs] = sum(kl_list) / len(kl_list)  # averaged over goals.
        avg_dist_for_valid_states_from_goals[vs] = avg_dist_of_vs_from_goals
        avg_total_reward_for_valid_states[vs] = kl_coeff * (avg_mi_for_valid_states[vs]) - dist_coeff * (
            avg_dist_for_valid_states_from_goals[vs]
        )

    visualize_metrics_map(avg_mi_for_valid_states, mdp_list[0], title="MI")
    visualize_metrics_map(avg_dist_for_valid_states_from_goals, mdp_list[0], title="DIST")
    visualize_metrics_map(avg_total_reward_for_valid_states, mdp_list[0], title="TOTAL REWARD")


def simulate_human_2d_modes_mdp():
    mdp_env_params = create_mdp_env_param_dict()
    mdp_list = create_mdp_list(mdp_env_params)

    # compute MI for all valid states
    p_vec = [1.0 / NUM_GOALS] * NUM_GOALS
    # p_vec = [0.5, 0.5, 0.0]
    # import IPython; IPython.embed(banner1='check ')
    print("V_exepcted")
    V_expected = visualize_V_expectation(mdp_list, prior=p_vec)
    import IPython

    IPython.embed(banner1="check V expected")
    compute_mi(mdp_list, mdp_env_params, p_vec)


if __name__ == "__main__":
    simulate_human_2d_modes_mdp()
    # simulate_turn_taking_and_inference()
    # check_vis_traj()
