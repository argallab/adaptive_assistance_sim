#!/usr/bin/env python

import pickle
import collections
import itertools
import random
import argparse
import copy
import numpy as np
import os

from mdp.mdp_discrete_SE2_gridworld_with_modes import MDPDiscreteSE2GridWorldWithModes
from mdp.mdp_utils import *
from adaptive_assistance_sim_utils import *

NUM_GOALS = [3, 4]
ALGO_CONDITIONS = ["disamb", "control"]
START_LOCATIONS = ["tr", "tl", "bl", "br"]
START_MODES = ["x", "y", "t"]
# maybe reconsider grid height and width
GRID_WIDTH = 10
GRID_HEIGHT = 10
NUM_ORIENTATIONS = 8
OCCUPANCY_LEVEL = 0.0
SPARSITY_FACTOR = 0.0
RAND_DIRECTION_FACTOR = 0.1


TOTAL_TRIALS = 48
TRIALS_PER_ALGO = TOTAL_TRIALS / len(ALGO_CONDITIONS)
NUM_BLOCKS_PER_ALGO = 3
TRIALS_PER_BLOCK_PER_ALGO = TRIALS_PER_ALGO / NUM_BLOCKS_PER_ALGO

START_REGION_WIDTH = 3
START_REGION_HEIGHT = 3

GOAL_REGION_WIDTH = 6
GOAL_REGION_HEIGHT = 4

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
    start_orientation = np.random.randint(NUM_ORIENTATIONS)
    start_mode = DIM_TO_MODE_INDEX[start_mode] + 1
    return (sampled_start_location[0], sampled_start_location[1], start_orientation, start_mode)


def create_random_goals_within_regions(num_goals, goal_region):
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


def _create_mdp_env_params_dict(start_location, num_goals):
    mdp_env_params = collections.OrderedDict()
    mdp_env_params["rl_algo_type"] = RlAlgoType.ValueIteration
    mdp_env_params["gamma"] = 0.96
    mdp_env_params["grid_width"] = GRID_WIDTH
    mdp_env_params["grid_height"] = GRID_HEIGHT
    mdp_env_params["num_discrete_orientations"] = NUM_ORIENTATIONS
    mdp_env_params["robot_type"] = CartesianRobotType.SE2
    mdp_env_params["mode_set_type"] = ModeSetType.OneD
    mdp_env_params["original_mdp_obstacles"] = []
    dynamics_obs_specs = []

    # make the list a tuple
    goal_region = GOAL_REGIONS_FOR_START_LOCATIONS[start_location]
    goal_list = create_random_goals_within_regions(num_goals=num_goals, goal_region=goal_region)

    for i, g in enumerate(goal_list):
        g = list(g)
        g.append(np.random.randint(mdp_env_params["num_discrete_orientations"]))
        goal_list[i] = tuple(g)

    mdp_env_params["all_goals"] = goal_list
    mdp_env_params["obstacle_penalty"] = -100
    mdp_env_params["goal_reward"] = 100
    mdp_env_params["step_penalty"] = -10
    mdp_env_params["sparsity_factor"] = SPARSITY_FACTOR
    mdp_env_params["rand_direction_factor"] = RAND_DIRECTION_FACTOR
    mdp_env_params["mdp_obstacles"] = []
    mdp_env_params["dynamic_obs_specs"] = dynamics_obs_specs

    return mdp_env_params


def create_mdp_list(mdp_env_params):
    mdp_list = []
    for i, g in enumerate(mdp_env_params["all_goals"]):
        mdp_env_params["mdp_goal_state"] = g
        # 2d goals.
        goals_that_are_obs = [(g_obs[0], g_obs[1]) for g_obs in mdp_env_params["all_goals"] if g_obs != g]
        mdp_env_params["mdp_obstacles"] = copy.deepcopy(mdp_env_params["original_mdp_obstacles"])
        mdp_env_params["mdp_obstacles"].extend(goals_that_are_obs)
        discrete_se2_modes_mdp = MDPDiscreteSE2GridWorldWithModes(copy.deepcopy(mdp_env_params))

        mdp_list.append(discrete_se2_modes_mdp)

    return mdp_list


def _convert_discrete_state_to_continuous_pose(discrete_state, cell_size, world_bounds):
    x_coord = discrete_state[0]
    y_coord = discrete_state[1]
    theta_coord = discrete_state[2]
    mode = discrete_state[3] - 1  # minus one because the dictionary is 0-indexed

    robot_position = [
        x_coord * cell_size["x"] + cell_size["x"] / 2.0 + world_bounds["xrange"]["lb"],
        y_coord * cell_size["y"] + cell_size["y"] / 2.0 + world_bounds["yrange"]["lb"],
    ]
    robot_orientation = (theta_coord * 2 * PI) / NUM_ORIENTATIONS
    start_mode = MODE_INDEX_TO_DIM[mode]

    return robot_position, robot_orientation, start_mode


def _generate_continuous_goal_poses(discrete_goal_list, cell_size, world_bounds):
    # could be moved outside
    goal_poses = []
    for dg in discrete_goal_list:
        goal_pose = [0, 0, 0]
        goal_pose[0] = (dg[0] * cell_size["x"]) + cell_size["x"] / 2.0 + world_bounds["xrange"]["lb"]
        goal_pose[1] = (dg[1] * cell_size["y"]) + cell_size["y"] / 2.0 + world_bounds["yrange"]["lb"]
        goal_pose[2] = (float(dg[2]) / NUM_ORIENTATIONS) * 2 * PI
        goal_poses.append(goal_pose)

    return goal_poses


def generate_experiment_trials(args):
    trial_dir = args.trial_dir
    metadata_dir = args.metadata_dir
    if not os.path.exists(trial_dir):
        os.makedirs(trial_dir)

    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)

    index = 0
    algo_condition_to_pkl_index = collections.defaultdict(list)
    world_bounds = collections.OrderedDict()
    world_bounds["xrange"] = collections.OrderedDict()
    world_bounds["yrange"] = collections.OrderedDict()
    # bottom left corner in continuous space
    world_bounds["xrange"]["lb"] = 0.05 * VIEWPORT_W / SCALE
    world_bounds["yrange"]["lb"] = 0.05 * VIEWPORT_H / SCALE
    world_bounds["xrange"]["ub"] = 0.75 * VIEWPORT_W / SCALE
    world_bounds["yrange"]["ub"] = 0.9 * VIEWPORT_H / SCALE
    for algo_condition, num_goals, start_location, start_mode in itertools.product(
        ALGO_CONDITIONS, NUM_GOALS, START_LOCATIONS, START_MODES
    ):
        trial_info_dict = collections.OrderedDict()
        trial_info_dict["world_bounds"] = world_bounds
        mdp_env_params = _create_mdp_env_params_dict(start_location, num_goals)
        # _init_goal_pfields_obstacles(mdp_env_params['dynamic_obs_specs'])
        mdp_env_params["cell_size"] = collections.OrderedDict()

        # create mdp list here. Select start positoin from valid stats.
        # generate continuous world bounds from width and height and cell size, and offset info
        mdp_env_params["cell_size"]["x"] = (
            world_bounds["xrange"]["ub"] - world_bounds["xrange"]["lb"]
        ) / mdp_env_params["grid_width"]
        mdp_env_params["cell_size"]["y"] = (
            world_bounds["yrange"]["ub"] - world_bounds["yrange"]["lb"]
        ) / mdp_env_params["grid_height"]
        trial_info_dict["all_mdp_env_params"] = mdp_env_params

        mdp_list = create_mdp_list(trial_info_dict["all_mdp_env_params"])
        trial_info_dict["mdp_list"] = mdp_list
        trial_info_dict["num_goals"] = num_goals
        trial_info_dict["is_visualize_grid"] = False

        discrete_robot_state = _create_start_state(start_location, start_mode)
        # discrete_robot_state = (0, 0, 0, 1)  # bottom left
        robot_position, robot_orientation, start_mode = _convert_discrete_state_to_continuous_pose(
            discrete_robot_state, mdp_env_params["cell_size"], world_bounds
        )
        trial_info_dict["robot_position"] = robot_position
        trial_info_dict["robot_orientation"] = robot_orientation
        trial_info_dict["start_mode"] = start_mode
        trial_info_dict["robot_type"] = CartesianRobotType.SE2
        trial_info_dict["mode_set_type"] = ModeSetType.OneD
        trial_info_dict["mode_transition_type"] = ModeTransitionType.Forward_Backward
        # generate continuous obstacle bounds based
        trial_info_dict["obstacles"] = []

        for o in mdp_env_params["original_mdp_obstacles"]:
            obs = collections.OrderedDict()
            obs["bottom_left"] = (
                o[0] * mdp_env_params["cell_size"]["x"] + world_bounds["xrange"]["lb"],
                o[1] * mdp_env_params["cell_size"]["y"] + world_bounds["yrange"]["lb"],
            )
            obs["top_right"] = (
                (o[0] + 1) * mdp_env_params["cell_size"]["x"] + world_bounds["xrange"]["lb"],
                (o[1] + 1) * mdp_env_params["cell_size"]["y"] + world_bounds["yrange"]["lb"],
            )
            trial_info_dict["obstacles"].append(obs)

        trial_info_dict["goal_poses"] = _generate_continuous_goal_poses(
            mdp_env_params["all_goals"], mdp_env_params["cell_size"], trial_info_dict["world_bounds"]
        )
        trial_info_dict["algo_condition"] = algo_condition
        trial_info_dict["spatial_window_half_length"] = 3
        trial_info_dict["kl_coeff"] = 0.9
        trial_info_dict["dist_coeff"] = 0.1
        with open(os.path.join(trial_dir, str(index) + ".pkl"), "wb") as fp:
            pickle.dump(trial_info_dict, fp)

        algo_condition_to_pkl_index[algo_condition].append(index)
        index += 1
        print("Trial Index ", index)
        print("               ")

    with open(os.path.join(metadata_dir, "algo_condition_to_pkl_index.pkl"), "wb") as fp:
        pickle.dump(algo_condition_to_pkl_index, fp)


def generate_turn_taking_practice_trials(args):
    turntaking_trial_dir = args.turntaking_trial_dir
    turntaking_metadata_dir = args.turntaking_metadata_dir
    if not os.path.exists(turntaking_trial_dir):
        os.makedirs(turntaking_trial_dir)

    if not os.path.exists(turntaking_metadata_dir):
        os.makedirs(turntaking_metadata_dir)

    index = 0
    world_bounds = collections.OrderedDict()
    world_bounds["xrange"] = collections.OrderedDict()
    world_bounds["yrange"] = collections.OrderedDict()
    # bottom left corner in continuous space
    world_bounds["xrange"]["lb"] = 0.05 * VIEWPORT_W / SCALE
    world_bounds["yrange"]["lb"] = 0.05 * VIEWPORT_H / SCALE
    world_bounds["xrange"]["ub"] = 0.75 * VIEWPORT_W / SCALE
    world_bounds["yrange"]["ub"] = 0.9 * VIEWPORT_H / SCALE

    for algo_condition in ["disamb", "control"]:
        for i in range(3):
            trial_info_dict = collections.OrderedDict()
            trial_info_dict["world_bounds"] = world_bounds
            start_location = random.choice(START_LOCATIONS)
            num_goals = random.choice(NUM_GOALS)
            start_mode = random.choice(START_MODES)

            mdp_env_params = _create_mdp_env_params_dict(start_location, num_goals)
            # _init_goal_pfields_obstacles(mdp_env_params['dynamic_obs_specs'])
            mdp_env_params["cell_size"] = collections.OrderedDict()

            # create mdp list here. Select start positoin from valid stats.
            # generate continuous world bounds from width and height and cell size, and offset info
            mdp_env_params["cell_size"]["x"] = (
                world_bounds["xrange"]["ub"] - world_bounds["xrange"]["lb"]
            ) / mdp_env_params["grid_width"]
            mdp_env_params["cell_size"]["y"] = (
                world_bounds["yrange"]["ub"] - world_bounds["yrange"]["lb"]
            ) / mdp_env_params["grid_height"]
            trial_info_dict["all_mdp_env_params"] = mdp_env_params

            mdp_list = create_mdp_list(trial_info_dict["all_mdp_env_params"])
            trial_info_dict["mdp_list"] = mdp_list
            trial_info_dict["num_goals"] = random.choice(range(3, 5))
            trial_info_dict["is_visualize_grid"] = False

            discrete_robot_state = _create_start_state(start_location, start_mode)
            # discrete_robot_state = (0, 0, 0, 1)  # bottom left
            robot_position, robot_orientation, start_mode = _convert_discrete_state_to_continuous_pose(
                discrete_robot_state, mdp_env_params["cell_size"], world_bounds
            )
            trial_info_dict["robot_position"] = robot_position
            trial_info_dict["robot_orientation"] = robot_orientation
            trial_info_dict["start_mode"] = start_mode
            trial_info_dict["robot_type"] = CartesianRobotType.SE2
            trial_info_dict["mode_set_type"] = ModeSetType.OneD
            trial_info_dict["mode_transition_type"] = ModeTransitionType.Forward_Backward
            # generate continuous obstacle bounds based
            trial_info_dict["obstacles"] = []
            for o in mdp_env_params["original_mdp_obstacles"]:
                obs = collections.OrderedDict()
                obs["bottom_left"] = (
                    o[0] * mdp_env_params["cell_size"]["x"] + world_bounds["xrange"]["lb"],
                    o[1] * mdp_env_params["cell_size"]["y"] + world_bounds["yrange"]["lb"],
                )
                obs["top_right"] = (
                    (o[0] + 1) * mdp_env_params["cell_size"]["x"] + world_bounds["xrange"]["lb"],
                    (o[1] + 1) * mdp_env_params["cell_size"]["y"] + world_bounds["yrange"]["lb"],
                )
                trial_info_dict["obstacles"].append(obs)

            trial_info_dict["goal_poses"] = _generate_continuous_goal_poses(
                mdp_env_params["all_goals"], mdp_env_params["cell_size"], trial_info_dict["world_bounds"]
            )
            trial_info_dict["algo_condition"] = algo_condition
            trial_info_dict["spatial_window_half_length"] = 3
            trial_info_dict["kl_coeff"] = 0.9
            trial_info_dict["dist_coeff"] = 0.1
            with open(os.path.join(turntaking_trial_dir, str(index) + ".pkl"), "wb") as fp:
                pickle.dump(trial_info_dict, fp)

            index += 1
            print("Trial Index ", index)
            print("               ")


def generate_training_trials(args):
    training_trial_dir = args.training_trial_dir
    training_metadata_dir = args.training_metadata_dir
    if not os.path.exists(training_trial_dir):
        os.makedirs(training_trial_dir)

    if not os.path.exists(training_metadata_dir):
        os.makedirs(training_metadata_dir)

    index = 0
    # algo_condition_to_pkl_index = collections.defaultdict(list)
    world_bounds = collections.OrderedDict()
    world_bounds["xrange"] = collections.OrderedDict()
    world_bounds["yrange"] = collections.OrderedDict()
    # bottom left corner in continuous space
    world_bounds["xrange"]["lb"] = 0.05 * VIEWPORT_W / SCALE
    world_bounds["yrange"]["lb"] = 0.05 * VIEWPORT_H / SCALE
    world_bounds["xrange"]["ub"] = 0.75 * VIEWPORT_W / SCALE
    world_bounds["yrange"]["ub"] = 0.9 * VIEWPORT_H / SCALE

    for blend_mode in ["blending", "teleop"]:
        for i in range(3):
            trial_info_dict = collections.OrderedDict()
            trial_info_dict["world_bounds"] = world_bounds
            start_location = random.choice(START_LOCATIONS)
            num_goals = random.choice(NUM_GOALS)
            start_mode = random.choice(START_MODES)

            mdp_env_params = _create_mdp_env_params_dict(start_location, num_goals)
            # _init_goal_pfields_obstacles(mdp_env_params['dynamic_obs_specs'])
            mdp_env_params["cell_size"] = collections.OrderedDict()

            # create mdp list here. Select start positoin from valid stats.
            # generate continuous world bounds from width and height and cell size, and offset info
            mdp_env_params["cell_size"]["x"] = (
                world_bounds["xrange"]["ub"] - world_bounds["xrange"]["lb"]
            ) / mdp_env_params["grid_width"]
            mdp_env_params["cell_size"]["y"] = (
                world_bounds["yrange"]["ub"] - world_bounds["yrange"]["lb"]
            ) / mdp_env_params["grid_height"]
            trial_info_dict["all_mdp_env_params"] = mdp_env_params

            mdp_list = create_mdp_list(trial_info_dict["all_mdp_env_params"])
            trial_info_dict["mdp_list"] = mdp_list
            trial_info_dict["num_goals"] = num_goals
            trial_info_dict["is_visualize_grid"] = False
            trial_info_dict["blend_mode"] = blend_mode

            discrete_robot_state = _create_start_state(start_location, start_mode)
            # discrete_robot_state = (0, 0, 0, 1)  # bottom left
            robot_position, robot_orientation, start_mode = _convert_discrete_state_to_continuous_pose(
                discrete_robot_state, mdp_env_params["cell_size"], world_bounds
            )
            trial_info_dict["robot_position"] = robot_position
            trial_info_dict["robot_orientation"] = robot_orientation
            trial_info_dict["start_mode"] = start_mode
            trial_info_dict["robot_type"] = CartesianRobotType.SE2
            trial_info_dict["mode_set_type"] = ModeSetType.OneD
            trial_info_dict["mode_transition_type"] = ModeTransitionType.Forward_Backward
            # generate continuous obstacle bounds based
            trial_info_dict["obstacles"] = []

            for o in mdp_env_params["original_mdp_obstacles"]:
                obs = collections.OrderedDict()
                obs["bottom_left"] = (
                    o[0] * mdp_env_params["cell_size"]["x"] + world_bounds["xrange"]["lb"],
                    o[1] * mdp_env_params["cell_size"]["y"] + world_bounds["yrange"]["lb"],
                )
                obs["top_right"] = (
                    (o[0] + 1) * mdp_env_params["cell_size"]["x"] + world_bounds["xrange"]["lb"],
                    (o[1] + 1) * mdp_env_params["cell_size"]["y"] + world_bounds["yrange"]["lb"],
                )
                trial_info_dict["obstacles"].append(obs)

            trial_info_dict["goal_poses"] = _generate_continuous_goal_poses(
                mdp_env_params["all_goals"], mdp_env_params["cell_size"], trial_info_dict["world_bounds"]
            )
            trial_info_dict["spatial_window_half_length"] = 3
            trial_info_dict["kl_coeff"] = 0.9
            trial_info_dict["dist_coeff"] = 0.1
            with open(os.path.join(training_trial_dir, str(index) + ".pkl"), "wb") as fp:
                pickle.dump(trial_info_dict, fp)

            index += 1
            print("Trial Index ", index)
            print("               ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trial_dir",
        dest="trial_dir",
        default=os.path.join(os.getcwd(), "trial_folders", "trial_dir"),
        help="The directory where trials will be stored are",
    )
    parser.add_argument(
        "--metadata_dir",
        dest="metadata_dir",
        default=os.path.join(os.getcwd(), "trial_folders", "metadata_dir"),
        help="The directory where metadata of trials will be stored",
    )
    parser.add_argument(
        "--training_trial_dir",
        dest="training_trial_dir",
        default=os.path.join(os.getcwd(), "trial_folders", "training_trial_dir"),
        help="The directory where trials will be stored are",
    )
    parser.add_argument(
        "--training_metadata_dir",
        dest="training_metadata_dir",
        default=os.path.join(os.getcwd(), "trial_folders", "training_metadata_dir"),
        help="The directory where metadata of trials will be stored",
    )
    parser.add_argument(
        "--turntaking_trial_dir",
        dest="turntaking_trial_dir",
        default=os.path.join(os.getcwd(), "trial_folders", "turntaking_trial_dir"),
        help="The directory where turn taking practice trials will be stored are",
    )
    parser.add_argument(
        "--turntaking_metadata_dir",
        dest="turntaking_metadata_dir",
        default=os.path.join(os.getcwd(), "trial_folders", "turntaking_metadata_dir"),
        help="The directory where metadata of turn taking trials will be stored",
    )
    parser.add_argument(
        "--num_reps_per_condition",
        action="store",
        type=int,
        default=1,
        help="number of repetetions for single combination of conditions ",
    )

    args = parser.parse_args()
    generate_experiment_trials(args)
    # generate_training_trials(args)
    generate_turn_taking_practice_trials(args)
