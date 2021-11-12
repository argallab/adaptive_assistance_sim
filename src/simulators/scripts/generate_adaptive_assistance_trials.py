import pickle
import collections
import itertools
import random
import argparse

import os

from mdp.mdp_discrete_SE2_gridworld_with_modes import MDPDiscreteSE2GridWorldWithModes
from mdp.mdp_utils import *
from adaptive_assistance_sim_utils import *

NUM_GOALS = [3, 4]
ALGO_CONDITIONS = ["disamb", "control"]
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


def generate_experiment_trials(args):
    trial_dir = args.trial_dir
    metadata_dir = args.metadata_dir
    if not os.path.exists(trial_dir):
        os.makedirs(trial_dir)

    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)


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
        "--num_reps_per_condition",
        action="store",
        type=int,
        default=1,
        help="number of repetetions for single combination of conditions ",
    )

    args = parser.parse_args()
    generate_experiment_trials(args)
