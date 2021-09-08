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
from mdp.mdp_discrete_2d_gridworld_with_modes import MDPDiscrete2DGridWorldWithModes
from adaptive_assistance_sim_utils import *
from mdp.mdp_utils import *
import matplotlib.pyplot as plt

# from inference_engine.goal_inferen import IntentInference

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
OCCUPANCY_LEVEL = 0.1
MAX_PATCHES = 4

GRID_WIDTH = 15
GRID_HEIGHT = 30
ENTROPY_THRESHOLD = 0.6
SPATIAL_WINDOW_HALF_LENGTH = 3


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
    mdp_env_params["original_mdp_obstacles"] = create_random_obstacles(
        width=mdp_env_params["grid_width"],
        height=mdp_env_params["grid_height"],
        occupancy_measure=OCCUPANCY_LEVEL,
        num_obstacle_patches=num_patches,
    )

    # mdp_env_params["original_mdp_obstacles"] = []
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


def compute_mi(mdp_list, mdp_env_params, prior, states_for_disamb_computation=None):
    if states_for_disamb_computation is None:
        states_for_disamb_computation = mdp_list[0].get_all_state_coords()

    avg_mi_for_valid_states = collections.OrderedDict()
    avg_dist_for_valid_states_from_goals = collections.OrderedDict()
    avg_total_reward_for_valid_states = collections.OrderedDict()
    num_trajectories = 50
    assert len(prior) == NUM_GOALS

    kl_coeff = 0.8
    dist_coeff = 0.2
    for i, vs in enumerate(states_for_disamb_computation):
        if i % 100 == 0:
            print("Computing MI for ", vs)
        # trajectories for each candidate state.
        traj_list = collections.defaultdict(list)
        for i in range(num_trajectories):
            sampled_goal_index = np.random.choice(NUM_GOALS, p=prior)  # sample a goal according to current belief
            mdp_for_sampled_goal = mdp_list[sampled_goal_index]
            # suboptimal a_sampled
            a_sampled = mdp_for_sampled_goal.get_optimal_action(vs, return_optimal=False)
            # sampled corrupted interface level action corresponding to task-level action, could be None
            phi = sample_phi_given_a(a_sampled)
            # corrupted interface level action, could be None
            phm = sample_phm_given_phi(phi)
            if phm != "None":
                applied_a = TRUE_INTERFACE_ACTION_TO_TASK_ACTION_MAP[phm]
            else:
                applied_a = "None"
            next_state = mdp_for_sampled_goal.get_next_state_from_state_action(vs, applied_a)
            traj_tuple = (vs, a_sampled, phi, phm, applied_a, next_state)
            traj_list[sampled_goal_index].append(traj_tuple)

        p_phm_g_s0 = collections.defaultdict(list)  # p(phm | g, s0)
        for g in traj_list.keys():
            for traj_g in traj_list[g]:
                (vs, a_sampled, phi, phm, applied_a, next_state) = traj_g
                p_phm_g_s0[g].append(INTERFACE_LEVEL_ACTIONS_TO_NUMBER_ID[phm])

        # p(a|s). is a list instead of defaultdict(list) because all actions are just combinaed
        p_phm_s0 = []
        for g in p_phm_g_s0.keys():
            p_phm_s0.extend(p_phm_g_s0[g])

        ph_actions_ids = INTERFACE_LEVEL_ACTIONS_TO_NUMBER_ID.values()

        p_phm_s0_hist = collections.Counter(p_phm_s0)
        # to make sure that all interface level actions are present in the histogram
        for ph_action_id in ph_actions_ids:
            if ph_action_id not in p_phm_s0_hist.keys():
                p_phm_s0_hist[ph_action_id] = 0

        p_phm_s = np.array(p_phm_s0_hist.values(), dtype=np.float32)
        p_phm_s = p_phm_s / np.sum(p_phm_s)
        kl_list = []

        for g in p_phm_g_s0.keys():
            p_phm_g_s_hist = collections.Counter(p_phm_g_s0[g])
            for ph_action_id in ph_actions_ids:
                if ph_action_id not in p_phm_g_s_hist.keys():
                    p_phm_g_s_hist[ph_action_id] = 0

            assert len(p_phm_g_s_hist) == len(p_phm_s)
            p_phm_g_s = np.array(p_phm_g_s_hist.values(), dtype=np.float32)
            p_phm_g_s = p_phm_g_s / np.sum(p_phm_g_s)
            kl = np.sum(special.rel_entr(p_phm_g_s, p_phm_s))
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
        avg_mi_for_valid_states[vs] = np.mean(kl_list)  # averaged over goals.
        avg_dist_for_valid_states_from_goals[vs] = avg_dist_of_vs_from_goals
        avg_total_reward_for_valid_states[vs] = kl_coeff * (avg_mi_for_valid_states[vs]) - dist_coeff * (
            avg_dist_for_valid_states_from_goals[vs]
        )

    visualize_metrics_map(avg_mi_for_valid_states, mdp_list[0], title="MI")
    visualize_metrics_map(avg_dist_for_valid_states_from_goals, mdp_list[0], title="DIST")
    visualize_metrics_map(avg_total_reward_for_valid_states, mdp_list[0], title="TOTAL REWARD")

    return avg_total_reward_for_valid_states


def sample_phi_given_a(a):  # sample from p(phii|a)
    global P_PHI_GIVEN_A, PHI_SPARSE_LEVEL
    d = np.random.rand()

    if d < PHI_SPARSE_LEVEL:
        phi = "None"
    else:
        p_vector = P_PHI_GIVEN_A[a].values()  # list of probabilities for phii
        # sample from the multinomial distribution with distribution p_vector
        phi_index_vector = np.random.multinomial(1, p_vector)
        phi_index = np.nonzero(phi_index_vector)[0][0]  # grab the index of the index_vector which had a nonzero entry
        phi = P_PHI_GIVEN_A[a].keys()[phi_index]  # retrieve phii using the phi_index
        # will be not None

    return phi


def sample_phm_given_phi(phi):  # sample from p(phm|phi)
    global P_PHM_GIVEN_PHI, PHM_SPARSE_LEVEL
    d = np.random.rand()
    if phi != "None":
        if d < PHM_SPARSE_LEVEL:
            phm = "None"
        else:
            p_vector = P_PHM_GIVEN_PHI[phi].values()  # list of probabilities for phm given phi
            phm_index_vector = np.random.multinomial(1, p_vector)  # sample from the multinomial distribution
            # grab the index of the index_vector which had a nonzero entry
            phm_index = np.nonzero(phm_index_vector)[0][0]
            phm = P_PHM_GIVEN_PHI[phi].keys()[phm_index]  # retrieve phm
    else:
        print("Sampled phi is None, therefore phm is None")
        phm = "None"

    return phm


def init_P_PHI_GIVEN_A():
    """
    Generate a random p(phi | a). key = a, subkey = ui
    """
    global P_PHI_GIVEN_A
    for k in TRUE_TASK_ACTION_TO_INTERFACE_ACTION_MAP.keys():  # task level action
        P_PHI_GIVEN_A[k] = collections.OrderedDict()
        for u in INTERFACE_LEVEL_ACTIONS:
            if u == TRUE_TASK_ACTION_TO_INTERFACE_ACTION_MAP[k]:
                # try to weight the true command more for realistic purposes. Can be offset by using a high PHI_GIVEN_A_NOISE
                P_PHI_GIVEN_A[k][u] = 1.0
            else:
                # P_PHI_GIVEN_A[k][u] = np.random.random()*PHI_GIVEN_A_NOISE #IF PHI_GIVEN_A_NOISE is 0, then the p(ui|a) is a deterministic mapping
                P_PHI_GIVEN_A[k][u] = 0.0

        delta_dist = np.array(P_PHI_GIVEN_A[k].values())
        uniform_dist = (1.0 / len(INTERFACE_LEVEL_ACTIONS)) * np.ones(len(INTERFACE_LEVEL_ACTIONS))
        blended_dist = (1 - PHI_GIVEN_A_NOISE) * delta_dist + PHI_GIVEN_A_NOISE * uniform_dist  # np.array
        for index, u in enumerate(INTERFACE_LEVEL_ACTIONS):
            P_PHI_GIVEN_A[k][u] = blended_dist[index]


def init_P_PHM_GIVEN_PHI():
    """
    Generates a random p(um|ui). key = ui, subkey = um
    """
    global P_PHM_GIVEN_PHI
    for i in INTERFACE_LEVEL_ACTIONS:  # ui
        P_PHM_GIVEN_PHI[i] = collections.OrderedDict()
        for j in INTERFACE_LEVEL_ACTIONS:  # um
            if i == j:
                # try to weight the true command more for realistic purposes. Can be offset by using a high UM_GIVEN_UI_NOISE
                P_PHM_GIVEN_PHI[i][j] = 1.0
            else:
                # P_PHM_GIVEN_PHI[i][j] = np.random.random()*UM_GIVEN_UI_NOISE#IF UM_GIVEN_UI_NOISE is 0, then the p(um|ui) is a deterministic mapping
                P_PHM_GIVEN_PHI[i][j] = 0.0

        delta_dist = np.array(P_PHM_GIVEN_PHI[i].values())
        uniform_dist = (1.0 / len(INTERFACE_LEVEL_ACTIONS)) * np.ones(len(INTERFACE_LEVEL_ACTIONS))
        blended_dist = (1 - PHM_GIVEN_PHI_NOISE) * delta_dist + PHM_GIVEN_PHI_NOISE * uniform_dist  # np.array
        for index, j in enumerate(INTERFACE_LEVEL_ACTIONS):
            P_PHM_GIVEN_PHI[i][j] = blended_dist[index]


def simulate_human_2d_modes_mdp():
    mdp_env_params = create_mdp_env_param_dict()
    mdp_list = create_mdp_list(mdp_env_params)

    # init interface noise levels
    init_P_PHI_GIVEN_A()
    init_P_PHM_GIVEN_PHI()

    # compute MI for all valid states
    p_vec = [1.0 / NUM_GOALS] * NUM_GOALS

    compute_mi(mdp_list, mdp_env_params, p_vec)


def _compute_entropy(p):
    uniform_distribution = np.array([1.0 / p.size] * p.size)
    max_entropy = entropy(uniform_distribution)
    normalized_entropy = entropy(p) / max_entropy
    return normalized_entropy


def simulate_turn_taking_and_inference():
    global P_PHI_GIVEN_A, P_PHM_GIVEN_PHI
    mdp_env_params = create_mdp_env_param_dict()
    mdp_list = []
    for i, g in enumerate(mdp_env_params["all_goals"]):
        mdp_env_params["mdp_goal_state"] = g
        goals_that_are_obs = [(g_obs[0], g_obs[1]) for g_obs in mdp_env_params["all_goals"] if g_obs != g]
        mdp_env_params["mdp_obstacles"] = copy.deepcopy(mdp_env_params["original_mdp_obstacles"])
        mdp_env_params["mdp_obstacles"].extend(goals_that_are_obs)
        discrete_2d_modes_mdp = MDPDiscrete2DGridWorldWithModes(copy.deepcopy(mdp_env_params))

        visualize_grid(discrete_2d_modes_mdp)
        # visualize_V_and_policy(discrete_2d_modes_mdp)
        mdp_list.append(discrete_2d_modes_mdp)

    # init interface noise levels
    init_P_PHI_GIVEN_A()
    init_P_PHM_GIVEN_PHI()

    ii_engine_params = collections.OrderedDict()
    ii_engine_params["mdps_for_goals"] = mdp_list
    ii_engine_params["p_phi_given_a"] = copy.deepcopy(P_PHI_GIVEN_A)
    ii_engine_params["p_phm_given_phi"] = copy.deepcopy(P_PHM_GIVEN_PHI)
    ii_engine_params["inference_type"] = "bayes_interface"
    ii_engine_params["use_state_information"] = False

    ii_engine = IntentInference(ii_engine_params)
    # initialize inference engine with proper params

    current_state = mdp_list[0].get_random_valid_state()  # pick a random state to start off with
    horizon = 100  # max num time steps

    # potentially change number of turns to be adaptive.
    # Only start turning when in vicinity of a zone which contains disambiguating states
    num_steps_per_turn = 3  # for human
    steps_per_turn_counter = 0

    # sample a goal according to current belief. Pick the true mdp or the true goal that the human is try to go for.
    # We are assuming that the human is going
    # to act according the MDP in an eps-optimal way
    p_vec = [1.0 / NUM_GOALS] * NUM_GOALS  # uninformed prior over goals
    g = np.random.choice(NUM_GOALS, p=p_vec)
    print("ALL GOALS ", mdp_env_params["all_goals"])
    print("RANDOM GOAL CHOICE ", g, mdp_env_params["all_goals"][g])
    print("Starting state", current_state)
    print("Starting belief", ii_engine.get_current_p_g_given_phm())
    previous_belief = ii_engine.get_current_p_g_given_phm()
    for t in range(horizon):
        # sampled task level action from policy, noisy. a_sampled are strings. Could be None depending on sparsity_factor
        a_sampled = mdp_list[g].get_optimal_action(current_state, return_optimal=False)
        # sampled corrupted interface level action corresponding to task-level action, could be None
        phi = sample_phi_given_a(a_sampled)
        # corrupted interface level action, could be None
        phm = sample_phm_given_phi(phi)
        # package all necessary information for inference engine
        inference_info_dict = {}
        inference_info_dict["phm"] = phm
        inference_info_dict["state"] = current_state
        ii_engine.perform_inference(inference_info_dict)
        current_belief = ii_engine.get_current_p_g_given_phm()
        print("Current state, a_sampled, phi, phm, belief", current_state, a_sampled, phi, phm, current_belief)

        print("CHANGE IN ENTROPY, prev, curr", previous_belief, current_belief)
        print(_compute_entropy(current_belief) - _compute_entropy(previous_belief))
        previous_belief = copy.deepcopy(current_belief)

        a_applied = TRUE_INTERFACE_ACTION_TO_TASK_ACTION_MAP[phm]
        next_state = mdp_list[g].get_next_state_from_state_action(current_state, a_applied)
        if mdp_list[g].check_if_state_coord_is_goal_state(next_state):
            print("Reached goal")
            break

        steps_per_turn_counter += 1
        if steps_per_turn_counter >= num_steps_per_turn:
            steps_per_turn_counter = 0
            current_belief_entropy = _compute_entropy(current_belief)
            if current_belief_entropy > ENTROPY_THRESHOLD:
                states_for_disamb_computation = _compute_spatial_window_around_current_state(current_state, mdp_list)
                # disamb_metric_all_vs = compute_mi(mdp_list, mdp_env_params, current_belief)
                # global_disamb_state = list(disamb_metric_all_vs.keys())[np.argmax(disamb_metric_all_vs.values())]
                # print("MAX DISAMB STATE", global_disamb_state)
                disamb_metric_local_spatial_window = compute_mi(
                    mdp_list, mdp_env_params, current_belief, states_for_disamb_computation
                )
                local_max_disamb_state = list(disamb_metric_local_spatial_window.keys())[
                    np.argmax(disamb_metric_local_spatial_window.values())
                ]
                print("LOCAL DISAMB STATE", local_max_disamb_state)
                # next_state = max_disamb_state
                next_state = local_max_disamb_state
            else:
                import IPython

                IPython.embed(banner1="confidentt in goal, perform shared conttrol.")
        current_state = next_state


def _compute_spatial_window_around_current_state(current_state, mdp_list):
    current_grid_loc = np.array(current_state[0:2])
    states_in_local_spatial_window = []

    all_state_coords = mdp_list[0].get_all_state_coords()
    window_coordinates = itertools.product(
        range(-SPATIAL_WINDOW_HALF_LENGTH + 1, SPATIAL_WINDOW_HALF_LENGTH),
        range(-SPATIAL_WINDOW_HALF_LENGTH + 1, SPATIAL_WINDOW_HALF_LENGTH),
    )
    for wc in window_coordinates:
        vs = current_grid_loc + np.array(wc)
        for mode in range(2):  # for 2d
            vs_mode = (vs[0], vs[1], mode + 1)
            if vs_mode in all_state_coords:
                states_in_local_spatial_window.append(vs_mode)

    return states_in_local_spatial_window


if __name__ == "__main__":
    simulate_human_2d_modes_mdp()
    # simulate_turn_taking_and_inference()
    # check_vis_traj()
