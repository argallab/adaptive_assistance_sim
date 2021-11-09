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
        import IPython

        IPython.embed(banner1="check data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-id", help="subject id", type=str)
    args = parser.parse_args()
    pphia = PhiGivenAAnalysis(args)
