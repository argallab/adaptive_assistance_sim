# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

import os
import pickle
import collections
import argparse
import random


def chunks(l, n):
    """
    code taken from https://chrisalbon.com/python/data_wrangling/break_list_into_chunks_of_equal_size/
    """
    for i in range(0, len(l), n):
        yield l[i : i + n]


def create_experiment_blocks(args):
    metadata_dir = args.metadata_dir
    metadata_filename = "algo_condition_to_pkl_index.pkl"
    subject_id = args.subject_id
    num_blocks = args.num_blocks

    assert os.path.exists(os.path.join(metadata_dir, metadata_filename))
    with open(os.path.join(metadata_dir, metadata_filename), "rb") as fp:
        algo_condition_to_pkl_index = pickle.load(fp)

    total_num_trials = args.total_num_trials  # TODO remove hardcoding
    assert len(algo_condition_to_pkl_index["control"]) == len(algo_condition_to_pkl_index["disamb"])
    num_trials_per_condition = len(algo_condition_to_pkl_index["control"])
    num_trials_per_block = total_num_trials / num_blocks

    #
    disamb_condition_list = algo_condition_to_pkl_index["disamb"]
    random.shuffle(disamb_condition_list)
    disamb_condition_blocks = list(chunks(disamb_condition_list, num_trials_per_block))
    for i, disamb_condition_block in enumerate(disamb_condition_blocks):
        filename = subject_id + "_disamb_condition_" + str(i) + "_num_blocks_" + str(num_blocks) + ".pkl"
        with open(os.path.join(metadata_dir, filename), "wb") as fp:
            pickle.dump(disamb_condition_block, fp)

    control_condition_list = algo_condition_to_pkl_index["control"]
    random.shuffle(control_condition_list)
    control_condition_blocks = list(chunks(control_condition_list, num_trials_per_block))
    for i, control_condition_block in enumerate(control_condition_blocks):
        filename = subject_id + "_control_condition_" + str(i) + "_num_blocks_" + str(num_blocks) + ".pkl"
        with open(os.path.join(metadata_dir, filename), "wb") as fp:
            pickle.dump(control_condition_block, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata_dir",
        dest="metadata_dir",
        default=os.path.join(os.getcwd(), "trial_folders", "metadata_dir"),
        help="The directory where metadata of trials will be stored",
    )
    parser.add_argument("--subject_id", dest="subject_id", default="deepak", help="unique_identifier for subject")
    parser.add_argument("--num_blocks", dest="num_blocks", default=6, help="total number of blocks")
    parser.add_argument("--total_num_trials", dest="total_num_trials", default=48, help="total number of trials")

    args = parser.parse_args()
    create_experiment_blocks(args)
