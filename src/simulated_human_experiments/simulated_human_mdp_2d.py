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

sys.path.append(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "inference_engine", "scripts"))


from mdp.mdp_discrete_2d_gridworld import MDPDiscrete2DGridWorld
from mdp.mdp_discrete_2d_gridworld_with_modes import MDPDiscrete2DGridWorldWithModes
from intent_inference import IntentInference
from adaptive_assistance_sim_utils import *
from mdp.mdp_utils import *
import matplotlib.pyplot as plt
from disamb_algo.discrete_mi_disamb_algo_2d import DiscreteMIDisambAlgo2D

# # low level commands issued by the snp interface. hp = hard puff, hs= hard sip, sp = soft puff, ss = soft sip. Also the domain for ui and um
# INTERFACE_LEVEL_ACTIONS = ["hp", "hs", "sp", "ss"]
# # high level actions, move_p = move in positive direction, move_n = move in negative direction, mode_r = switch mode to right, mode_l = switch mode to left. positive and negative is conditioned on mode
# TASK_LEVEL_ACTIONS = ["move_p", "move_n", "to_mode_r", "to_mode_l"]
# # true mapping of a to phi
# TRUE_TASK_ACTION_TO_INTERFACE_ACTION_MAP = collections.OrderedDict(
#     {"move_p": "sp", "move_n": "ss", "to_mode_r": "hp", "to_mode_l": "hs"}
# )
# # true inverse mapping of phi to a
# TRUE_INTERFACE_ACTION_TO_TASK_ACTION_MAP = collections.OrderedDict(
#     {v: k for k, v in TRUE_TASK_ACTION_TO_INTERFACE_ACTION_MAP.items()}
# )

# INTERFACE_LEVEL_ACTIONS_TO_NUMBER_ID = {"sp": 0, "ss": 1, "hp": 2, "hs": 3}

# p(phii|a)
P_PHI_GIVEN_A = collections.OrderedDict()
# Lower the number, lower the error. Between 0 and 1. If 0, the p(ui|a) is delta and same as the true mapping
PHI_GIVEN_A_NOISE = 0.0

# p(phm|ui)
P_PHM_GIVEN_PHI = collections.OrderedDict()
PHM_GIVEN_PHI_NOISE = 0.0  # Lower the number, lower the error. Between 0 and 1. If 0, no difference between ui and um

PHI_SPARSE_LEVEL = 0.0
PHM_SPARSE_LEVEL = 0.0
MODE_DICT = {0: "Horizontal", 1: "Vertical"}

NUM_GOALS = 3
OCCUPANCY_LEVEL = 0.0
MAX_PATCHES = 4

GRID_WIDTH = 15
GRID_HEIGHT = 30
ENTROPY_THRESHOLD = 0.65
SPATIAL_WINDOW_HALF_LENGTH = 3


# HUMAN_DICTIONS

P_PHI_GIVEN_A_HUMAN = collections.OrderedDict()
DEFAULT_PHI_GIVEN_A_NOISE_HUMAN = 0.0
P_PHM_GIVEN_PHI_HUMAN = collections.OrderedDict()
DEFAULT_PHM_GIVEN_PHI_NOISE_HUMAN = 0.0

PHI_SPARSE_LEVEL_HUMAN = 0.0
PHM_SPARSE_LEVEL_HUMAN = 0.0

# DICTIONARIES
DIM_TO_MODE_INDEX_XYZ = {"x": 0, "y": 1, "z": 2, "gr": 3}
MODE_INDEX_TO_DIM_XYZ = {v: k for k, v in DIM_TO_MODE_INDEX.items()}

# region based goal generation

START_REGION_WIDTH = 6
START_REGION_HEIGHT = 8

GOAL_REGION_WIDTH = 10
GOAL_REGION_HEIGHT = 12
START_LOCATIONS = ["tr", "tl", "bl", "br"]

INTER_GOAL_THRESHOLD = 1

REGIONS_FOR_START_LOCATIONS = {
    "tr": {
        "region_bl": (GRID_WIDTH - START_REGION_WIDTH, GRID_HEIGHT - START_REGION_HEIGHT),
        "region_tr": (GRID_WIDTH - 1, GRID_HEIGHT - 1),
    },
    "tl": {"region_bl": (0, GRID_HEIGHT - START_REGION_HEIGHT), "region_tr": (START_REGION_WIDTH, GRID_HEIGHT - 1)},
    "bl": {"region_bl": (0, 0), "region_tr": (START_REGION_WIDTH, START_REGION_HEIGHT)},
    "br": {"region_bl": (GRID_WIDTH - START_REGION_WIDTH, 0), "region_tr": (GRID_WIDTH - 1, START_REGION_HEIGHT)},
}

GOAL_REGIONS_FOR_START_LOCATIONS = {
    "tr": {"region_bl": (0, 0), "region_tr": (GOAL_REGION_WIDTH, GOAL_REGION_HEIGHT)},
    "tl": {"region_bl": (GRID_WIDTH - GOAL_REGION_WIDTH, 0), "region_tr": (GRID_WIDTH - 1, GOAL_REGION_HEIGHT)},
    "bl": {
        "region_bl": (GRID_WIDTH - GOAL_REGION_WIDTH, GRID_HEIGHT - GOAL_REGION_HEIGHT),
        "region_tr": (GRID_WIDTH - 1, GRID_HEIGHT - 1),
    },
    "br": {"region_bl": (0, GRID_HEIGHT - GOAL_REGION_HEIGHT), "region_tr": (GOAL_REGION_WIDTH, GRID_HEIGHT - 1)},
}


def _create_start_state(start_location, start_mode):
    start_region = REGIONS_FOR_START_LOCATIONS[start_location]
    region_bl = start_region["region_bl"]
    region_tr = start_region["region_tr"]
    region_coords = list(itertools.product(range(region_bl[0], region_tr[0]), range(region_bl[1], region_tr[1])))
    sampled_start_location = random.sample(list(set(region_coords)), 1)[0]
    return (sampled_start_location[0], sampled_start_location[1], start_mode)


def create_random_goals_within_regions(goal_region, num_goals):
    region_bl = goal_region["region_bl"]
    region_tr = goal_region["region_tr"]
    region_coords = list(itertools.product(range(region_bl[0], region_tr[0]), range(region_bl[1], region_tr[1])))

    random_goals = []
    sampled_goal = random.sample(list(set(region_coords) - set(random_goals)), 1)[0]
    random_goals.append(sampled_goal)
    while len(random_goals) < num_goals:
        # tuple
        sampled_goal = random.sample(list(set(region_coords) - set(random_goals)), 1)[0]
        dist_to_goals = [np.linalg.norm(np.array(sampled_goal) - np.array(g)) for g in random_goals]
        if min(dist_to_goals) > INTER_GOAL_THRESHOLD:
            random_goals.append(sampled_goal)
        else:
            continue

    return random_goals


def create_mdp_env_param_dict(start_location, num_goals):
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
    if start_location == None:
        mdp_env_params["all_goals"] = [(3, 18), (8, 5), (14, 6)]
    else:
        goal_region = GOAL_REGIONS_FOR_START_LOCATIONS[start_location]
        goal_list = create_random_goals_within_regions(goal_region=goal_region, num_goals=num_goals)
        mdp_env_params["all_goals"] = goal_list
    #
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


def create_mdp_list(mdp_env_params, gen_autonomy=False):
    mdp_list = []
    if gen_autonomy:
        autonomy_mdp_list = []

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
        if gen_autonomy:
            discrete_2d_mdp = MDPDiscrete2DGridWorld(copy.deepcopy(mdp_env_params))
            autonomy_mdp_list.append(discrete_2d_mdp)

    if gen_autonomy:
        return mdp_list, autonomy_mdp_list
    else:
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
    min_val = min(metric_dict.values())
    grids = min_val * np.ones((mdp_params["num_modes"], mdp_params["grid_width"], mdp_params["grid_height"]))
    fig, ax = plt.subplots(1, mdp_params["num_modes"])
    for k in metric_dict.keys():
        grids[k[-1] - 1, k[0], k[1]] = metric_dict[k]
    for i in range(mdp_params["num_modes"]):
        grid = grids[i, :, :]
        grid = grid.T
        grid = np.flipud(grid)
        im = ax[i].imshow(grid)
        ax[i].set_title("D(s) - {} Mode".format(MODE_DICT[i]))
        cbar = ax[i].figure.colorbar(im, ax=ax[i])
    fig.tight_layout()
    plt.show()


def policy_for_all_mdps(mdp_list):

    fig, ax = plt.subplots(mdp_list[0].get_env_params()["num_modes"], len(mdp_list))
    for mdp_i, mdp in enumerate(mdp_list):
        mdp_params = mdp.get_env_params()
        _, policy_dict = mdp.get_optimal_policy_for_mdp()
        grids = np.zeros((mdp_params["num_modes"], mdp_params["grid_width"], mdp_params["grid_height"]))
        for s in policy_dict.keys():
            grids[s[-1] - 1, s[0], s[1]] = policy_dict[s]

        for i in range(mdp_params["num_modes"]):
            grid = grids[i, :, :]
            grid = grid.T
            grid = np.flipud(grid)
            im = ax[i, mdp_i].imshow(grid)
            ax[i, mdp_i].set_title("Learned Policy Goal {} - {} Mode".format(mdp_i + 1, MODE_DICT[i]))
            if mdp_i == len(mdp_list) - 1:
                cbar = ax[i, mdp_i].figure.colorbar(im, ax=ax[i, mdp_i])
                cbar.ax.set_ylabel(
                    "move_p,       move_n,      switch_r,      switch_l", rotation=90, va="bottom", size=12, labelpad=30
                )

    # fig.subplots_adjust(wspace=10, hspace=0)
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
        cbar.ax.set_ylabel("V", rotation=-90, va="bottom", labelpad=4)

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


def visualize_goals(mdp):
    mdp_params = mdp.get_env_params()
    all_goals = mdp_params["all_goals"]
    grid = np.zeros((mdp_params["grid_width"], mdp_params["grid_height"]))
    fig, ax = plt.subplots(1, 1)
    for g in all_goals:
        grid[g[0], g[1]] = 100

    grid = grid.T
    grid = np.flipud(grid)
    im = ax.imshow(grid)
    ax.set_title("Goal Configuration")
    fig.tight_layout()
    plt.show()


def _generate_continuous_goal_poses(discrete_goal_list, cell_size, world_bounds):
    goal_poses = []
    for dg in discrete_goal_list:
        goal_pose = [0, 0]
        goal_pose[0] = (dg[0] * cell_size["x"]) + cell_size["x"] / 2.0 + world_bounds["xrange"]["lb"]
        goal_pose[1] = (dg[1] * cell_size["y"]) + cell_size["y"] / 2.0 + world_bounds["yrange"]["lb"]
        goal_poses.append(goal_pose)

    return goal_poses


def generate_2d_disamb_heatmaps(prior=None):
    mdp_env_params = create_mdp_env_param_dict(start_location=None, num_goals=3)

    mdp_list = create_mdp_list(mdp_env_params, gen_autonomy=False)

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
    env_params["phi_given_a_noise"] = 0.001
    env_params["phm_given_phi_noise"] = 0.001

    env_params["goal_poses"] = _generate_continuous_goal_poses(
        mdp_env_params["all_goals"], mdp_env_params["cell_size"], world_bounds
    )
    env_params["mdp_list"] = mdp_list

    # policy_for_all_mdps(mdp_list)
    # for mdp in mdp_list:
    #     visualize_goals(mdp)
    #     visualize_V_and_policy(mdp)

    if prior == None:
        prior = [1 / NUM_GOALS] * NUM_GOALS
    disamb_algo = DiscreteMIDisambAlgo2D(env_params, "adnadnak")
    states_for_disamb_computation = mdp_list[0].get_all_state_coords_with_grid_locs_diff_from_goals_and_obs()
    continuous_positions_of_local_spatial_window = [
        convert_discrete_state_to_continuous_position(s, mdp_env_params["cell_size"], world_bounds)
        for s in states_for_disamb_computation
    ]
    disamb_algo._compute_mi(prior, states_for_disamb_computation, continuous_positions_of_local_spatial_window)

    visualize_metrics_map(disamb_algo.avg_mi_for_valid_states, mdp_list[0], title="MI")
    visualize_metrics_map(disamb_algo.dist_of_vs_from_weighted_mean_of_goals, mdp_list[0], title="WEIGHTED DIST")
    visualize_metrics_map(disamb_algo.avg_total_reward_for_valid_states, mdp_list[0], title="REWARD")


def _compute_entropy(p):
    uniform_distribution = np.array([1.0 / p.size] * p.size)
    max_entropy = entropy(uniform_distribution)
    normalized_entropy = entropy(p) / max_entropy
    return normalized_entropy


def simulate_turn_taking_only_teleop():
    global P_PHI_GIVEN_A_HUMAN, P_PHM_GIVEN_PHI_HUMAN, DEFAULT_PHI_GIVEN_A_NOISE_HUMAN, DEFAULT_PHM_GIVEN_PHI_NOISE_HUMAN, PHI_SPARSE_LEVEL_HUMAN, PHM_SPARSE_LEVEL_HUMAN

    DEFAULT_PHI_GIVEN_A_NOISE_HUMAN = 0.2
    DEFAULT_PHM_GIVEN_PHI_NOISE_HUMAN = 0.2
    PHI_SPARSE_LEVEL_HUMAN = 0.03
    PHM_SPARSE_LEVEL_HUMAN = 0.03

    num_trials = 10
    for trial in range(num_trials):

        start_location = random.choice(START_LOCATIONS)
        num_goals = random.choice([3, 4])
        print("START LOCATION, NUM_GOALS ", start_location, num_goals)
        mdp_env_params = create_mdp_env_param_dict(start_location, num_goals)
        mdp_list, autonomy_mdp_list = create_mdp_list(mdp_env_params, gen_autonomy=True)

        # world bounds and cell size
        world_bounds = collections.OrderedDict()
        world_bounds["xrange"] = collections.OrderedDict()
        world_bounds["yrange"] = collections.OrderedDict()
        # bottom left corner in continuous space
        world_bounds["xrange"]["lb"] = 0.05 * VIEWPORT_W / SCALE
        world_bounds["yrange"]["lb"] = 0.05 * VIEWPORT_H / SCALE
        world_bounds["xrange"]["ub"] = 0.75 * VIEWPORT_W / SCALE
        world_bounds["yrange"]["ub"] = 0.9 * VIEWPORT_H / SCALE

        mdp_env_params["cell_size"] = collections.OrderedDict()
        mdp_env_params["cell_size"]["x"] = (
            world_bounds["xrange"]["ub"] - world_bounds["xrange"]["lb"]
        ) / mdp_env_params["grid_width"]
        mdp_env_params["cell_size"]["y"] = (
            world_bounds["yrange"]["ub"] - world_bounds["yrange"]["lb"]
        ) / mdp_env_params["grid_height"]

        # init human interface params
        init_P_PHI_GIVEN_A_HUMAN()
        init_P_PHM_GIVEN_PHI_HUMAN()

        # init autonomous controller.

        # instantiate disamb algo
        env_params = collections.OrderedDict()
        env_params["all_mdp_env_params"] = mdp_env_params
        env_params["robot_type"] = CartesianRobotType.R2
        env_params["mode_set_type"] = ModeSetType.OneD
        env_params["spatial_window_half_length"] = 3
        env_params["phi_given_a_noise"] = 0.1
        env_params["phm_given_phi_noise"] = 0.1
        env_params["goal_poses"] = _generate_continuous_goal_poses(
            mdp_env_params["all_goals"], mdp_env_params["cell_size"], world_bounds
        )
        env_params["mdp_list"] = mdp_list
        env_params["kl_coeff"] = 0.5
        env_params["dist_coeff"] = 0.5

        disamb_algo = DiscreteMIDisambAlgo2D(env_params, "adnadnak")

        # instantiate intent inference
        ii_engine_params = collections.OrderedDict()
        ii_engine_params["mdps_for_goals"] = mdp_list
        # noise levels that atonomy THINKS human has.
        ii_engine_params["phi_given_a_noise"] = 0.1
        ii_engine_params["phm_given_phi_noise"] = 0.1

        ii_engine = IntentInference(ii_engine_params)

        # current_state = mdp_list[0].get_random_valid_state(is_not_goal=True)  # pick a random state to start off with

        start_mode = np.random.randint(2) + 1  # either 1 or 2
        current_state = _create_start_state(start_location, start_mode)
        horizon = 100
        min_num_steps_per_turn = 3
        max_num_steps_per_turn = 6  # for human
        steps_per_turn_counter = 0
        num_steps_per_turn = random.randint(min_num_steps_per_turn, max_num_steps_per_turn)
        p_vec = [1.0 / NUM_GOALS] * NUM_GOALS
        human_sampled_goal_id = np.random.choice(NUM_GOALS, p=p_vec)
        print("ALL GOALS ", mdp_env_params["all_goals"])
        print("RANDOM GOAL CHOICE ", human_sampled_goal_id, mdp_env_params["all_goals"][human_sampled_goal_id])
        print("Starting state", current_state)
        print("Starting belief", ii_engine.get_current_p_g_given_phm())
        previous_belief = ii_engine.get_current_p_g_given_phm()

        # instantiate autonomy policy. acting according to full 2D MDP without modes

        # DISAMB
        for t in range(horizon):
            # USE human_mdp
            if mdp_list[human_sampled_goal_id].check_if_state_coord_is_goal_state(current_state):
                print("Reached goal")
                break
            a_sampled = mdp_list[human_sampled_goal_id].get_optimal_action(current_state, return_optimal=False)
            # sampled corrupted interface level action corresponding to task-level action, could be None
            phi = sample_phi_given_a_human(a_sampled, current_mode=current_state[-1])
            # corrupted interface level action, could be None
            phm = sample_phm_given_phi_human(phi)
            # package all necessary information for inference engine
            inference_info_dict = {}
            inference_info_dict["phm"] = phm
            inference_info_dict["state"] = current_state

            ii_engine.perform_inference(inference_info_dict)
            current_belief = ii_engine.get_current_p_g_given_phm()
            print(" ")
            print("Current state, a_sampled, phi, phm, belief", current_state, a_sampled, phi, phm, current_belief)
            # print("CHANGE IN ENTROPY, prev, curr", previous_belief, current_belief)
            # print(_compute_entropy(current_belief) - _compute_entropy(previous_belief))
            previous_belief = copy.deepcopy(current_belief)
            if not phm == "None":
                a_applied = TRUE_INTERFACE_ACTION_TO_TASK_ACTION_MAP[phm]
            else:
                a_applied = "None"

            next_state = mdp_list[human_sampled_goal_id].get_next_state_from_state_action(current_state, a_applied)

            current_position = convert_discrete_state_to_continuous_position(
                current_state, mdp_env_params["cell_size"], world_bounds
            )
            steps_per_turn_counter += 1
            if steps_per_turn_counter >= num_steps_per_turn:
                steps_per_turn_counter = 0
                num_steps_per_turn = random.randint(min_num_steps_per_turn, max_num_steps_per_turn)
                current_belief_entropy = _compute_entropy(current_belief)
                if current_belief_entropy > ENTROPY_THRESHOLD:
                    # perform disamb. get max local disamb state.
                    # teleport to that state. Continue.
                    local_max_disamb_state = disamb_algo.get_local_disamb_state(
                        current_belief, current_state, current_position
                    )
                    print("DISAMB TRANSITION FROM TO", current_state, local_max_disamb_state)
                    next_state = local_max_disamb_state
                else:

                    inferred_goal_id = np.argmax(current_belief)

                    autonomy_steps = random.randint(min_num_steps_per_turn, max_num_steps_per_turn)
                    ss = current_state[0:2]
                    opt_traj = autonomy_mdp_list[inferred_goal_id].get_optimal_trajectory_from_state(
                        ss, horizon=autonomy_steps + 1, return_optimal=True
                    )
                    last_state = opt_traj[-1][-1]  # get last state from autonomy traj.
                    # set next state as the last state from autonomy ttraj with same mode.
                    next_state = (last_state[0], last_state[1], current_state[-1])
                    print("CONTROL TRANSITION FROM TO", current_state, next_state, opt_traj)

                    # get inferred goal.
                    # get full 2d policy for inferred goal. keep current mode.
                    # simulate for fixed steps. Get final state. append current_mode. Update state.
            current_state = next_state


# For human actions. The noise params can be different from what autonomy this the human is
def sample_phi_given_a_human(a, current_mode):  # sample from p(phii|a)
    global P_PHI_GIVEN_A_HUMAN, PHI_SPARSE_LEVEL_HUMAN
    d = np.random.rand()

    if d < PHI_SPARSE_LEVEL_HUMAN:
        phi = "None"
    else:
        p_vector = list(P_PHI_GIVEN_A_HUMAN[current_mode][a].values())  # list of probabilities for phii
        # sample from the multinomial distribution with distribution p_vector
        phi_index_vector = np.random.multinomial(1, p_vector)
        # grab the index of the index_vector which had a nonzero entry
        phi_index = np.nonzero(phi_index_vector)[0][0]
        phi = list(P_PHI_GIVEN_A_HUMAN[current_mode][a].keys())[phi_index]  # retrieve phii using the phi_index
        # will be not None

    return phi


def sample_phm_given_phi_human(phi):  # sample from p(phm|phi)
    global P_PHM_GIVEN_PHI, P_PHM_GIVEN_PHI_HUMAN
    d = np.random.rand()
    if phi != "None":
        if d < PHM_SPARSE_LEVEL_HUMAN:
            phm = "None"
        else:
            p_vector = list(P_PHM_GIVEN_PHI_HUMAN[phi].values())  # list of probabilities for phm given phi
            phm_index_vector = np.random.multinomial(1, p_vector)  # sample from the multinomial distribution
            # grab the index of the index_vector which had a nonzero entry
            phm_index = np.nonzero(phm_index_vector)[0][0]
            phm = list(P_PHM_GIVEN_PHI_HUMAN[phi].keys())[phm_index]  # retrieve phm
    else:
        print("Sampled phi is None, therefore phm is None")
        phm = "None"

    return phm


def init_P_PHI_GIVEN_A_HUMAN():
    # only to be done at the beginning of a session for a subject. No updating between trials
    global P_PHI_GIVEN_A_HUMAN, DEFAULT_PHI_GIVEN_A_NOISE_HUMAN
    P_PHI_GIVEN_A_HUMAN = collections.OrderedDict()
    for mode in [1, 2]:  # hard coded modes for SE2
        P_PHI_GIVEN_A_HUMAN[mode] = collections.OrderedDict()
        for k in TRUE_TASK_ACTION_TO_INTERFACE_ACTION_MAP.keys():  # task level action
            P_PHI_GIVEN_A_HUMAN[mode][k] = collections.OrderedDict()
            for u in INTERFACE_LEVEL_ACTIONS:
                if u == TRUE_TASK_ACTION_TO_INTERFACE_ACTION_MAP[k]:
                    # try to weight the true command more for realistic purposes. Can be offset by using a high PHI_GIVEN_A_NOISE
                    P_PHI_GIVEN_A_HUMAN[mode][k][u] = 1.0
                else:
                    P_PHI_GIVEN_A_HUMAN[mode][k][u] = 0.0

            delta_dist = np.array(list(P_PHI_GIVEN_A_HUMAN[mode][k].values()))
            uniform_dist = (1.0 / len(INTERFACE_LEVEL_ACTIONS)) * np.ones(len(INTERFACE_LEVEL_ACTIONS))
            blended_dist = (
                1 - DEFAULT_PHI_GIVEN_A_NOISE_HUMAN
            ) * delta_dist + DEFAULT_PHI_GIVEN_A_NOISE_HUMAN * uniform_dist  # np.array
            for index, u in enumerate(INTERFACE_LEVEL_ACTIONS):
                P_PHI_GIVEN_A_HUMAN[mode][k][u] = blended_dist[index]


def init_P_PHM_GIVEN_PHI_HUMAN():
    global P_PHM_GIVEN_PHI_HUMAN, DEFAULT_PHM_GIVEN_PHI_NOISE_HUMAN
    P_PHM_GIVEN_PHI_HUMAN = collections.OrderedDict()
    for i in INTERFACE_LEVEL_ACTIONS:  # ui
        P_PHM_GIVEN_PHI_HUMAN[i] = collections.OrderedDict()
        for j in INTERFACE_LEVEL_ACTIONS:  # um
            if i == j:
                # try to weight the true command more for realistic purposes. Can be offset by using a high UM_GIVEN_UI_NOISE
                P_PHM_GIVEN_PHI_HUMAN[i][j] = 1.0
            else:
                # P_PHM_GIVEN_PHI[i][j] = np.random.random()*UM_GIVEN_UI_NOISE#IF UM_GIVEN_UI_NOISE is 0, then the p(um|ui) is a deterministic mapping
                P_PHM_GIVEN_PHI_HUMAN[i][j] = 0.0

        delta_dist = np.array(list(P_PHM_GIVEN_PHI_HUMAN[i].values()))
        uniform_dist = (1.0 / len(INTERFACE_LEVEL_ACTIONS)) * np.ones(len(INTERFACE_LEVEL_ACTIONS))
        blended_dist = (
            1 - DEFAULT_PHM_GIVEN_PHI_NOISE_HUMAN
        ) * delta_dist + DEFAULT_PHM_GIVEN_PHI_NOISE_HUMAN * uniform_dist  # np.array
        for index, j in enumerate(INTERFACE_LEVEL_ACTIONS):
            P_PHM_GIVEN_PHI_HUMAN[i][j] = blended_dist[index]


if __name__ == "__main__":
    generate_2d_disamb_heatmaps(prior=[0.5, 0.5, 0])
    # simulate_turn_taking_only_teleop()
