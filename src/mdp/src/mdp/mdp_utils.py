import math
from enum import Enum
import collections
import itertools
import random
import numpy as np

PI = math.pi


class RlAlgoType(Enum):  # These are the different options available in the mdptoolbox
    ValueIteration = 0
    PolicyIteration = 1
    QLearning = 2
    PolicyIterationModified = 3
    ValueIterationGS = 4


class IntentInferenceType(Enum):
    DFT = 0
    RecursiveBayes = 1
    TrajectoryBayes = 2
    HeuristicConfidence = 3


class ControlType(Enum):
    Cartesian = 0
    Joint = 1


# class CartesianRobotType(Enum):
#     R2 = 2  # (x,y)
#     SE2_NH = 2  # (v, w)
#     R3 = 3  # (x, y, z)
#     SE2 = 3  # (x,y,theta)
#     SE3 = 6  # (x,y,z,r,p,yaw)


class JointRobotType(Enum):
    J1 = 1
    J2 = 2
    J3 = 3
    J4 = 4
    J5 = 5
    J6 = 6
    J7 = 7


class Dim(Enum):  # potential split the "state dimension" and "action dimensions" into different Enums
    X = 0
    VX = 0  # dimension of action for x dimension
    Y = 1
    VY = 1  # dimension of action for y dimension
    W = 0  # angular velocity dimension for nonholonomic system
    V = 1  # linear velocity dimension for NH system
    Mode2D = 2
    Theta = 2
    ModeSE2_NH = 3
    Mode3D = 3
    ModeSE2 = 3


class ControlIndicatorType(Enum):
    Bar = 0
    Circle = 1


# class ModeSetType(Enum):
#     OneD = 1
#     TwoD = 2
#     ThreeD = 3


class InterfaceType(Enum):
    Joystick = 0
    SNP = 1
    HA = 2


# class ModeTransitionType(Enum):
#     Direct = 0
#     Forward = 1
#     Backward = 2
#     Forward_Backward = 3


class WorldType(Enum):
    ContinuousR2 = 0
    GridWorld2DModes = 1


class TransitionType(Enum):
    VALID = 0
    INTO_OBSTACLE = 1
    INTO_WALL = 2
    INVALID = 3


# note: The RobotType determines the "max" dim number that will show up in ANY of the dict values.
# THe ModeSetType will be determine the max "length" of the value string.
# CARTESIAN_MODE_SET_OPTIONS = {
#     CartesianRobotType.R2: {ModeSetType.OneD: {1: 1, 2: 2}, ModeSetType.TwoD: {1: 12}},
#     CartesianRobotType.SE2_NH: {ModeSetType.OneD: {1: 1, 2: 2}, ModeSetType.TwoD: {1: 12}},
#     CartesianRobotType.R3: {
#         ModeSetType.OneD: {1: 1, 2: 2, 3: 3},
#         ModeSetType.TwoD: {1: 12, 2: 23, 3: 13},
#         ModeSetType.ThreeD: {1: 123},
#     },
#     CartesianRobotType.SE2: {
#         ModeSetType.OneD: {1: 1, 2: 2, 3: 3},
#         ModeSetType.TwoD: {1: 12, 2: 3},
#         ModeSetType.ThreeD: {1: 123},
#     },
#     CartesianRobotType.SE3: {},
# }

# CARTESIAN_DIM_NAMES = {
#     CartesianRobotType.R2: ["X", "Y"],
#     CartesianRobotType.SE2_NH: ["V", "W"],
#     CartesianRobotType.R3: ["X", "Y", "Z"],
#     CartesianRobotType.SE2: ["X", "Y", "YAW"],
#     CartesianRobotType.SE3: ["X", "Y", "Z", "ROLL", "PITCH", "YAW"],
# }

# CARTESIAN_DIM_LABELS = {
#     CartesianRobotType.R2: {"X": "L/R", "Y": "F/B"},
#     CartesianRobotType.SE2_NH: {"V": "F/B", "W": "L/R"},
#     CartesianRobotType.R3: {"X": "L/R", "Y": "F/B", "Z": "U/D"},
#     CartesianRobotType.SE2: {"X": "L/R", "Y": "F/B", "YAW": "YAW"},
#     CartesianRobotType.SE3: {"X": "L/R", "Y": "F/B", "Z": "U/D", "ROLL": "ROLL", "PITCH": "PITCH", "YAW": "YAW"},
# }

# CARTESIAN_DIM_TO_CTRL_INDEX_MAP = {
#     CartesianRobotType.R2: {"X": 0, "Y": 1},
#     CartesianRobotType.SE2_NH: {"V": 0, "W": 1},
#     CartesianRobotType.R3: {"X": 0, "Y": 1, "Z": 2},
#     CartesianRobotType.SE2: {"X": 0, "Y": 1, "YAW": 2},
#     CartesianRobotType.SE3: {"X": 0, "Y": 1, "Z": 2, "ROLL": 3, "PITCH": 4, "YAW": 5},
# }

# CARTESIAN_DIM_INDICATOR_TYPES = {
#     CartesianRobotType.R2: {"X": ControlIndicatorType.Bar, "Y": ControlIndicatorType.Bar},
#     CartesianRobotType.SE2_NH: {"V": ControlIndicatorType.Bar, "W": ControlIndicatorType.Circle},
#     CartesianRobotType.R3: {
#         "X": ControlIndicatorType.Bar,
#         "Y": ControlIndicatorType.Bar,
#         "Z": ControlIndicatorType.Bar,
#     },
#     CartesianRobotType.SE2: {
#         "X": ControlIndicatorType.Bar,
#         "Y": ControlIndicatorType.Bar,
#         "YAW": ControlIndicatorType.Circle,
#     },
#     CartesianRobotType.SE3: {
#         "X": ControlIndicatorType.Bar,
#         "Y": ControlIndicatorType.Bar,
#         "Z": ControlIndicatorType.Bar,
#         "ROLL": ControlIndicatorType.Circle,
#         "PITCH": ControlIndicatorType.Circle,
#         "YAW": ControlIndicatorType.Circle,
#     },
# }

# # the following can be efficienctly created using loops
# JOINT_DIM_NAMES = {
#     JointRobotType.J1: ["J1"],
#     JointRobotType.J2: ["J1", "J2"],
#     JointRobotType.J3: ["J1", "J2", "J3"],
#     JointRobotType.J4: ["J1", "J2", "J3", "J4"],
#     JointRobotType.J5: ["J1", "J2", "J3", "J4", "J5"],
#     JointRobotType.J6: ["J1", "J2", "J3", "J4", "J5", "J6"],
#     JointRobotType.J7: ["J1", "J2", "J3", "J4", "J5", "J6", "J7"],
# }

# JOINT_DIM_TO_CTRL_INDEX_MAP = {
#     JointRobotType.J1: {"J1": 0},
#     JointRobotType.J2: {"J1": 0, "J2": 1},
#     JointRobotType.J3: {"J1": 0, "J2": 1, "J3": 2},
#     JointRobotType.J4: {"J1": 0, "J2": 1, "J3": 2, "J4": 3},
#     JointRobotType.J5: {"J1": 0, "J2": 1, "J3": 2, "J4": 3, "J5": 4},
#     JointRobotType.J6: {"J1": 0, "J2": 1, "J3": 2, "J4": 3, "J5": 4, "J6": 5},
#     JointRobotType.J7: {"J1": 0, "J2": 1, "J3": 2, "J4": 3, "J5": 4, "J6": 5, "J7": 6},
# }

# JOINT_DIM_INDICATOR_TYPES = {
#     JointRobotType.J1: {"J1": ControlIndicatorType.Circle},
#     JointRobotType.J2: {"J1": ControlIndicatorType.Circle, "J2": ControlIndicatorType.Circle},
#     JointRobotType.J3: {
#         "J1": ControlIndicatorType.Circle,
#         "J2": ControlIndicatorType.Circle,
#         "J3": ControlIndicatorType.Circle,
#     },
#     JointRobotType.J4: {
#         "J1": ControlIndicatorType.Circle,
#         "J2": ControlIndicatorType.Circle,
#         "J3": ControlIndicatorType.Circle,
#         "J4": ControlIndicatorType.Circle,
#     },
#     JointRobotType.J5: {
#         "J1": ControlIndicatorType.Circle,
#         "J2": ControlIndicatorType.Circle,
#         "J3": ControlIndicatorType.Circle,
#         "J4": ControlIndicatorType.Circle,
#         "J5": ControlIndicatorType.Circle,
#     },
#     JointRobotType.J6: {
#         "J1": ControlIndicatorType.Circle,
#         "J2": ControlIndicatorType.Circle,
#         "J3": ControlIndicatorType.Circle,
#         "J4": ControlIndicatorType.Circle,
#         "J5": ControlIndicatorType.Circle,
#         "J6": ControlIndicatorType.Circle,
#     },
#     JointRobotType.J7: {
#         "J1": ControlIndicatorType.Circle,
#         "J2": ControlIndicatorType.Circle,
#         "J3": ControlIndicatorType.Circle,
#         "J4": ControlIndicatorType.Circle,
#         "J5": ControlIndicatorType.Circle,
#         "J6": ControlIndicatorType.Circle,
#         "J7": ControlIndicatorType.Circle,
#     },
# }

START_DIST_THRESHOLD = 2
INTER_GOAL_THRESHOLD = 2


def create_random_obstacles(width, height, occupancy_measure, num_obstacle_patches, dirichlet_scale_param=10):
    assert occupancy_measure < 1.0 and occupancy_measure >= 0.0
    num_cells = width * height
    num_occupied_cells = int(round(occupancy_measure * num_cells))
    num_cells_for_all_patches = list(
        np.int32(
            np.round(num_occupied_cells * np.random.dirichlet(np.ones(num_obstacle_patches) * dirichlet_scale_param))
        )
    )

    all_cell_coords = list(itertools.product(range(width), range(height)))
    # pick three random starting points
    obstacle_patch_seeds = random.sample(all_cell_coords, num_obstacle_patches)

    def get_random_obstacle_neighbors(obs):
        def check_bounds(state):
            state[0] = max(0, min(state[0], width - 1))
            state[1] = max(0, min(state[1], height - 1))
            return state

        top_neighbor = tuple(check_bounds(np.array(obs) + (0, 1)))
        bottom_neighbor = tuple(check_bounds(np.array(obs) + (0, -1)))
        left_neighbor = tuple(check_bounds(np.array(obs) + (-1, 0)))
        right_neighbor = tuple(check_bounds(np.array(obs) + (1, 0)))

        all_neighbors = [top_neighbor, bottom_neighbor, left_neighbor, right_neighbor]
        # return all_neighbors
        num_neighbors_to_be_returned = random.randint(1, len(all_neighbors))
        return random.sample(all_neighbors, num_neighbors_to_be_returned)

    obstacle_list = []
    for i, (num_cells_for_patch, patch_seed) in enumerate(zip(num_cells_for_all_patches, obstacle_patch_seeds)):
        # print('Creating obstacle patch ', i)
        obstacles_in_patch = [tuple(patch_seed)]
        while len(obstacles_in_patch) <= num_cells_for_patch:
            new_cells = []
            for obs in obstacles_in_patch:
                new_cells.extend(get_random_obstacle_neighbors(obs))
            obstacles_in_patch.extend(new_cells)
            obstacles_in_patch = list(set(obstacles_in_patch))  # remove duplicates

        obstacle_list.extend(obstacles_in_patch)

    return obstacle_list


def create_random_goals(width, height, num_goals, obstacle_list):
    all_cell_coords = list(itertools.product(range(width), range(height)))
    random_goals = []
    sampled_goal = random.sample(list(set(all_cell_coords) - set(obstacle_list) - set(random_goals)), 1)[0]
    random_goals.append(sampled_goal)  # add the first goal into the array.
    # print(random_goals)
    while len(random_goals) < num_goals:
        sampled_goal = random.sample(list(set(all_cell_coords) - set(obstacle_list) - set(random_goals)), 1)[0]  # tuple
        dist_to_goals = [np.linalg.norm(np.array(sampled_goal) - np.array(g)) for g in random_goals]
        if min(dist_to_goals) > INTER_GOAL_THRESHOLD:
            random_goals.append(sampled_goal)
        else:
            continue

    return random_goals


# Modify the following with **kwargs to deal with R2, R2modes, SE2 and SE2Modes


def create_random_start_state(width, height, obstacle_list, goal_list, mode_set):
    all_cell_coords = list(itertools.product(range(width), range(height)))
    dist_to_goals = [-1000]
    while min(dist_to_goals) < START_DIST_THRESHOLD:
        random_start_state = random.sample(list(set(all_cell_coords) - set(goal_list) - set(obstacle_list)), 1)[0]
        dist_to_goals = [np.linalg.norm(np.array(random_start_state) - np.array(g)) for g in goal_list]

    random_mode = random.sample(mode_set.keys(), 1)  # [m]

    return tuple(list(random_start_state) + random_mode)  # a tuple
