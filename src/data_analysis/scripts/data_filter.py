#!/usr/bin/env python

import os
import csv
import sys
import argparse
import pandas as pd
import copy
import pickle
import collections


class DataFilter(object):
    def __init__(self, args):

        self.condition = "control"
        self._extract_testing_data()

    def _extract_testing_data(self):
        print("Extracting Test Data")
        self.testing_metadata_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "simulators",
            "scripts",
            "trial_folders",
            "metadata_dir",
        )
        self.testing_trial_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "simulators",
            "scripts",
            "trial_folders",
            "trial_dir",
        )
        self.data_analysis_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data_analysis", "all_data"
        )  # this is where the unified dict will eventually be saved.
        self.unified_testing_data_dict_path = os.path.join(
            self.data_analysis_dir, "unified_testing_data_dict_filename" + str(self.condition) + ".pkl"
        )
        self.unified_testing_data_dict = None
        with open(self.unified_testing_data_dict_path, "rb") as fp:
            if sys.version_info.major == 3:
                self.unified_testing_data_dict = pickle.load(fp, encoding="latin1")
            else:  # if python 2
                self.unified_testing_data_dict = pickle.load(fp)
        assert self.unified_testing_data_dict is not None
        self.testing_trial_metadata_dict_path = os.path.join(
            self.testing_metadata_dir, "algo_condition_to_pkl_index.pkl"
        )
        self.testing_trial_metadata_dict = None

        with open(self.testing_trial_metadata_dict_path, "rb") as fp:
            if sys.version_info.major == 3:
                self.testing_trial_metadata_dict = pickle.load(fp, encoding="latin1")
            else:
                self.testing_trial_metadata_dict = pickle.load(fp)
        assert self.testing_trial_metadata_dict is not None

        self.all_testing_data_indexed_by_condition = collections.defaultdict(list)
        # self.all_testing_data_indexed_by_subject = collections.defaultdict(list)
        self.all_testing_data_indexed_by_subject = collections.OrderedDict()
        self.all_testing_data_indexed_by_block = collections.defaultdict(list)
        for _, trial_dict in self.unified_testing_data_dict.items():
            trial_index_for_trial = trial_dict["trial_index_for_trial"]
            subject_id = trial_dict["subject_id"]
            if subject_id not in self.all_testing_data_indexed_by_subject:
                self.all_testing_data_indexed_by_subject[subject_id] = collections.defaultdict(list)
            block_id = trial_dict["block_id"]
            # find the key in self.testing_trial_metadata_dict whose list of indices contain trial_index_for_trial
            condition_for_trial = [
                condition
                for condition, index_list_for_condition in self.testing_trial_metadata_dict.items()
                if trial_index_for_trial in index_list_for_condition
            ]
            assert len(condition_for_trial) == 1  # there can only be one condition associated with a trial
            self.all_testing_data_indexed_by_condition[condition_for_trial[0]].append(trial_dict)
            # self.all_testing_data_indexed_by_subject[subject_id].append(trial_dict)
            self.all_testing_data_indexed_by_subject[subject_id][condition_for_trial[0]].append(trial_dict)
            self.all_testing_data_indexed_by_block[block_id].append(trial_dict)


if __name__ == "__main__":
    dobj = DataFilter()
