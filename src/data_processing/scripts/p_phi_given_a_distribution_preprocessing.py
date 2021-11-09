#!/usr/bin/env python
# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

import os
import csv
import sys
import argparse
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import itertools
import collections
import bisect
import rospkg

sys.path.append(os.path.join(rospkg.RosPack().get_path("simulators"), "scripts"))
from adaptive_assistance_sim_utils import TRUE_ACTION_TO_COMMAND, LOW_LEVEL_COMMANDS


class DataParser(object):
    def __init__(self, file_dir):
        super(DataParser, self).__init__()

        results_files = os.listdir(file_dir)
        action_prompt_file = os.path.join(file_dir, "_action_prompt.csv")
        user_response_file = os.path.join(file_dir, "_user_response.csv")

        self.action_prompt_df = self.read_csv_files(action_prompt_file)
        self.user_response_df = self.read_csv_files(user_response_file)

    def read_csv_files(self, file_path):

        df = pd.read_csv(file_path, header=0)
        return df


class PhiGivenAAnalysis(object):
    def __init__(self, args):

        self.id = args.id
        self.file_dir = os.path.join(
            os.path.abspath(os.path.join(os.getcwd(), os.pardir)), "raw_data", self.id + "_p_phi_given_a"
        )
        if not os.path.exists(self.file_dir):
            os.makedirs(self.file_dir)

        self.data = DataParser(self.file_dir)

    def get_nearest_time_stamp(self, tq, time_stamp_array):
        """
        assumes that time_stamp_array is a sorted (ascending) list.
        returns: (time stamp closest to tq, index of the time stamp closest to tq in time_stamp_array)
        """
        # idx is the index in the time_stamp_array whose value is immediately greater than tq
        idx = bisect.bisect(time_stamp_array, tq)
        # if requested timestamp is smaller than the first element of the list then return the first element of the list and the index is 0
        if idx == 0:
            return time_stamp_array[idx], idx
        elif idx == len(time_stamp_array):
            # if tq is greater than the biggest timestamp in the list, then retirn the last timestamp and the index of len-1 or idx-1
            return time_stamp_array[idx - 1], idx - 1
        else:
            prev_t = time_stamp_array[idx - 1]
            next_t = time_stamp_array[idx]
            if next_t - tq <= tq - prev_t:  # if tq is closer to the next value
                return time_stamp_array[idx], idx
            else:  # tq is closer to the previous value
                return time_stamp_array[idx - 1], idx - 1

    def get_user_response_block_indices(self, time_s, time_e, user_input):

        assert "rosbagTimestamp" in user_input
        time_s_u, index_of_time_s_u = self.get_nearest_time_stamp(time_s, user_input.rosbagTimestamp)
        time_e_u, index_of_time_e_u = self.get_nearest_time_stamp(time_e, user_input.rosbagTimestamp)

        if index_of_time_e_u == index_of_time_s_u:  # sanity check
            user_response_block_indices = index_of_time_s_u

        elif time_s_u < time_s and time_e_u < time_e:
            user_response_block_indices = index_of_time_e_u

        elif time_s_u > time_s and time_e_u > time_e:
            user_response_block_indices = index_of_time_s_u

        else:
            user_response_block_indices = []  # no user response (it's from previous)

        return user_response_block_indices

    def build_distributions(self):

        # create dictionary to map user input to the low level commands for the final p_ui_given_a dict
        USER_RESPONSE_DICT = {"1": "Hard Puff", "2": "Hard Sip", "3": "Soft Puff", "4": "Soft Sip"}

        # hard puff, hard sip, soft puff, soft sip
        up = np.zeros(4)
        down = np.zeros(4)
        right = np.zeros(4)
        left = np.zeros(4)
        ccw = np.zeros(4)
        cw = np.zeros(4)
        mode_r_x = np.zeros(4)
        mode_r_y = np.zeros(4)
        mode_r_t = np.zeros(4)
        mode_l_x = np.zeros(4)
        mode_l_y = np.zeros(4)
        mode_l_t = np.zeros(4)

        # because there are x2 as many prmpts (the actual command and the empty command after)
        # and 12 total actions, so the remaining is the number of times each action was shown since all shown equally
        iters_per_action = len(self.data.action_prompt_df) / 24

        # dictionary for mapping action prompts to the arrays we want to fill
        ACTION_TO_ARRAY_DICT = {
            "up": up,
            "down": down,
            "left": left,
            "right": right,
            "clockwise": cw,
            "counterclockwise": ccw,
            "mode_switch_right_1": mode_r_x,
            "mode_switch_right_2": mode_r_y,
            "mode_switch_right_3": mode_r_t,
            "mode_switch_left_1": mode_l_x,
            "mode_switch_left_2": mode_l_y,
            "mode_switch_left_3": mode_l_t,
        }

        # keep dict of lengths, to reduce normalizer if person missed input
        ACTION_TO_ARRAY_NORMALIZER_DICT = {
            "up": iters_per_action,
            "down": iters_per_action,
            "left": iters_per_action,
            "right": iters_per_action,
            "clockwise": iters_per_action,
            "counterclockwise": iters_per_action,
            "mode_switch_right_1": iters_per_action,
            "mode_switch_right_2": iters_per_action,
            "mode_switch_right_3": iters_per_action,
            "mode_switch_left_1": iters_per_action,
            "mode_switch_left_2": iters_per_action,
            "mode_switch_left_3": iters_per_action,
        }

        # for every action prompt, get the user response (if they did respond)
        for i in range(0, len(self.data.action_prompt_df) - 1, 2):

            key = self.data.action_prompt_df.at[i, "command"].replace('"', "")

            prompt_t_s = self.data.action_prompt_df.at[i, "rosbagTimestamp"]
            prompt_t_e = self.data.action_prompt_df.at[i + 1, "rosbagTimestamp"]

            user_response_block_indices = self.get_user_response_block_indices(
                prompt_t_s, prompt_t_e, self.data.user_response_df
            )

            if user_response_block_indices != []:  # if they gave a response
                user_response = int(
                    self.data.user_response_df["command"][user_response_block_indices].replace('"', "")
                )
                ACTION_TO_ARRAY_DICT[key][user_response - 1] += 1
            else:
                ACTION_TO_ARRAY_NORMALIZER_DICT[key] -= 1

        # normalize
        for k, v in ACTION_TO_ARRAY_DICT.items():
            v = v / ACTION_TO_ARRAY_NORMALIZER_DICT[k]
            ACTION_TO_ARRAY_DICT[k] = v

        import IPython

        IPython.embed(banner1="check distrib")

        self.create_p_phi_given_a(ACTION_TO_ARRAY_DICT)

    def create_p_phi_given_a(self, probabilities):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-id", help="subject id", type=str)
    args = parser.parse_args()
    pphia = PhiGivenAAnalysis(args)
    pphia.build_distributions()
