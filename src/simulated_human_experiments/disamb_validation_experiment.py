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
        mdp_env_params["all_goals"] = [(3, 18), (8, 5), (10, 13)]  # , (14, 6), (12, 1)]
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


def _generate_continuous_goal_poses(discrete_goal_list, cell_size, world_bounds):
    goal_poses = []
    for dg in discrete_goal_list:
        goal_pose = [0, 0]
        goal_pose[0] = (dg[0] * cell_size["x"]) + cell_size["x"] / 2.0 + world_bounds["xrange"]["lb"]
        goal_pose[1] = (dg[1] * cell_size["y"]) + cell_size["y"] / 2.0 + world_bounds["yrange"]["lb"]
        goal_poses.append(goal_pose)

    return goal_poses


def _create_world():
    global P_PHI_GIVEN_A_HUMAN, P_PHM_GIVEN_PHI_HUMAN, DEFAULT_PHI_GIVEN_A_NOISE_HUMAN, DEFAULT_PHM_GIVEN_PHI_NOISE_HUMAN, PHI_SPARSE_LEVEL_HUMAN, PHM_SPARSE_LEVEL_HUMAN

    DEFAULT_PHI_GIVEN_A_NOISE_HUMAN = 0.0
    DEFAULT_PHM_GIVEN_PHI_NOISE_HUMAN = 0.0
    PHI_SPARSE_LEVEL_HUMAN = 0.0
    PHM_SPARSE_LEVEL_HUMAN = 0.0

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

    # init human interface params
    init_P_PHI_GIVEN_A_HUMAN()
    init_P_PHM_GIVEN_PHI_HUMAN()

    env_params = collections.OrderedDict()
    env_params["all_mdp_env_params"] = mdp_env_params
    env_params["robot_type"] = CartesianRobotType.R2
    env_params["mode_set_type"] = ModeSetType.OneD
    env_params["spatial_window_half_length"] = 3
    env_params["phi_given_a_noise"] = 0.000
    env_params["phm_given_phi_noise"] = 0.000
    env_params["kl_coeff"] = 1.0
    env_params["dist_coeff"] = 0.0

    env_params["goal_poses"] = _generate_continuous_goal_poses(
        mdp_env_params["all_goals"], mdp_env_params["cell_size"], world_bounds
    )
    env_params["mdp_list"] = mdp_list

    # instantiate intent inference
    ii_engine_params = collections.OrderedDict()
    ii_engine_params["mdps_for_goals"] = mdp_list
    # noise levels that atonomy THINKS human has.
    ii_engine_params["phi_given_a_noise"] = 0.0
    ii_engine_params["phm_given_phi_noise"] = 0.0

    ii_engine = IntentInference(ii_engine_params)

    return mdp_env_params, mdp_list, world_bounds, env_params, ii_engine


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


def convert_discrete_state_to_continuous_position(discrete_state, cell_size, world_bounds):
    x_coord = discrete_state[0]
    y_coord = discrete_state[1]

    position = [
        x_coord * cell_size["x"] + cell_size["x"] / 2.0 + world_bounds["xrange"]["lb"],
        y_coord * cell_size["y"] + cell_size["y"] / 2.0 + world_bounds["yrange"]["lb"],
    ]

    return position


def validate_disamb_metric():
    global P_PHI_GIVEN_A_HUMAN, P_PHM_GIVEN_PHI_HUMAN, DEFAULT_PHI_GIVEN_A_NOISE_HUMAN, DEFAULT_PHM_GIVEN_PHI_NOISE_HUMAN, PHI_SPARSE_LEVEL_HUMAN, PHM_SPARSE_LEVEL_HUMAN

    DEFAULT_PHI_GIVEN_A_NOISE_HUMAN = 0.0
    DEFAULT_PHM_GIVEN_PHI_NOISE_HUMAN = 0.0
    PHI_SPARSE_LEVEL_HUMAN = 0.00
    PHM_SPARSE_LEVEL_HUMAN = 0.00

    num_trials = 10
    for trial in range(num_trials):

        mdp_env_params, mdp_list, world_bounds, env_params, ii_engine = _create_world()

        prior = [1 / NUM_GOALS] * NUM_GOALS
        disamb_algo = DiscreteMIDisambAlgo2D(env_params, "adnadnak")
        states_for_disamb_computation = mdp_list[0].get_all_state_coords_with_grid_locs_diff_from_goals_and_obs()
        continuous_positions_of_local_spatial_window = [
            convert_discrete_state_to_continuous_position(s, mdp_env_params["cell_size"], world_bounds)
            for s in states_for_disamb_computation
        ]
        # disamb_algo._compute_mi(prior, states_for_disamb_computation, continuous_positions_of_local_spatial_window)

        # visualize_metrics_map(disamb_algo.avg_total_reward_for_valid_states, mdp_list[0], title="REWARD")

        for s in states_for_disamb_computation:
            # compute local neightbohood for candidate sttaes.
            continuous_position_for_s = convert_discrete_state_to_continuous_position(
                s, mdp_env_params["cell_size"], world_bounds
            )

            (
                max_disamb_state,
                states_in_local_spatial_window,
                continuous_positions_of_local_spatial_window,
            ) = disamb_algo.get_local_disamb_state(prior, s, continuous_position_for_s)

            ground_truth_s = collections.OrderedDict()
            for cs in states_in_local_spatial_window:
                s_disamb_ground_truth = collections.defaultdict(list)
                for a in mdp_list[0].task_level_actions.keys():
                    ii_engine.reset_belief()
                    start_belief = ii_engine.get_current_p_g_given_phm()
                    # sampled corrupted interface level action (a) corresponding to task-level action, could be None
                    phi = sample_phi_given_a_human(a, current_mode=cs[-1])
                    # corrupted interface level action, could be None
                    phm = sample_phm_given_phi_human(phi)
                    # package all necessary information for inference engine
                    inference_info_dict = {}
                    inference_info_dict["phm"] = phm
                    inference_info_dict["state"] = cs
                    ii_engine.perform_inference(inference_info_dict)
                    current_belief = ii_engine.get_current_p_g_given_phm()

                    sorted_belief = sorted(current_belief)
                    # difference between max and second max probabilities.
                    s_disamb_ground_truth[a] = sorted_belief[-1] - sorted_belief[-2]

                ground_truth_s[cs] = np.mean(np.array(list(s_disamb_ground_truth.values())))


            import IPython

            IPython.embed(banner1="check reward")
            break
            # for each of the candidate states. compute the average prob gain for all actions.

        # ground_truth_s = collections.OrderedDict()
        # for s in states_for_disamb_computation:
        #     s_disamb_ground_truth = collections.defaultdict(list)
        #     for a in mdp_list[0].task_level_actions.keys():
        #         ii_engine.reset_belief()
        #         start_belief = ii_engine.get_current_p_g_given_phm()
        #         # sampled corrupted interface level action (a) corresponding to task-level action, could be None
        #         phi = sample_phi_given_a_human(a, current_mode=s[-1])
        #         # corrupted interface level action, could be None
        #         phm = sample_phm_given_phi_human(phi)
        #         # package all necessary information for inference engine
        #         inference_info_dict = {}
        #         inference_info_dict["phm"] = phm
        #         inference_info_dict["state"] = s
        #         ii_engine.perform_inference(inference_info_dict)
        #         current_belief = ii_engine.get_current_p_g_given_phm()
        #         # print(" ")
        #         # print("Current state, a_sampled, phi, phm, belief", s, a, phi, phm, start_belief, current_belief)

        #         sorted_belief = sorted(current_belief)
        #         # difference between max and second max probabilities.
        #         s_disamb_ground_truth[a] = sorted_belief[-1] - sorted_belief[-2]

        #     ground_truth_s[s] = np.mean(np.array(list(s_disamb_ground_truth.values())))

        import IPython

        IPython.embed(banner1="chck")
        break


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
    validate_disamb_metric()
    # simulate_turn_taking_only_teleop()
