#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append(os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)), "simulators/scripts/"))
import itertools
import scipy.stats as ss
import statsmodels.api as sa
import scikit_posthocs as sp
import collections

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
# matplotlib.rcParams["text.usetex"] = True

RED_GOAL_INDEX = 0
COLOR_LIST = {"control": "red", "disamb": "blue"}


class CompareAssistanceParadigms(object):
    def __init__(self, args):

        self.directory_list = list()
        self.directory_path = args.directory_path
        self.metrics = args.metrics

        self.data_dir = os.path.join(self.directory_path, "parsed_trial_data")
        print(self.data_dir)
        self.get_recursive_folders()
        self.labels = ["Control", "Disambiguation"]
        self.assistance_cond = ["control", "disamb"]
        self.label_to_plot_pos = {"Control": 0, "Disambiguation": 1}
        self.v_strong_alpha = 0.001
        self.strong_alpha = 0.01
        self.alpha = 0.05

    def get_recursive_folders(self):
        for root, dirs, files in os.walk(self.data_dir, topdown=False):
            for name in dirs:
                self.directory_list.append(os.path.join(root, name))

    def load_trial_data(self):
        # get list of all trial csv files for all subject folders in the parsed_trial_data directory
        self.files_name_list = []
        self.files_path_list = []
        for folder in self.directory_list:
            [self.files_name_list.append(f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
            [
                self.files_path_list.append(os.path.join(folder, f))
                for f in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, f))
            ]

    # helper functions
    def autonomy_turn_markers_per_trial(self, df):
        # assumes that the autonomy was the first agent to move always.
        autonomy_turn_start_indices = df["turn_indicator"][
            df["turn_indicator"].str.contains("autonomy") == True
        ].index.tolist()
        autonomy_turn_end_indices = df[df["turn_indicator"] == "human"].index.tolist()[1:]
        return autonomy_turn_start_indices, autonomy_turn_end_indices

    def human_turn_markers_per_trial(self, df):
        human_turn_start_indices = df[df["turn_indicator"] == "human"].index.tolist()
        human_turn_end_indices = df["turn_indicator"][
            df["turn_indicator"].str.contains("autonomy") == True
        ].index.tolist()

        trial_end_index_list = df[df["trial_marker"] == "end"].index.tolist()
        assert len(trial_end_index_list) == 1
        human_turn_end_indices.append(trial_end_index_list[0])

        return human_turn_start_indices, human_turn_end_indices

    def algo_computation_markers_per_trial(self, df):
        algo_computation_start_indices = df[df["assistance_compute_marker"] == "before"].index.tolist()
        algo_computation_end_indices = df[df["assistance_compute_marker"] == "after"].index.tolist()

        return algo_computation_start_indices, algo_computation_end_indices

    # metric computations
    def compute_num_turns(self, df):
        autonomy_turn_start_indices, autonomy_turn_end_indices = self.autonomy_turn_markers_per_trial(df)
        human_turn_start_indices, human_turn_end_indices = self.human_turn_markers_per_trial(df)

        total_human_turn_time_for_trial = 0.0
        for human_turn_counter, (turn_start_ind, turn_end_ind) in enumerate(
            zip(human_turn_start_indices, human_turn_end_indices)
        ):
            total_human_turn_time_for_trial += df["time"][turn_end_ind] - df["time"][turn_start_ind]

        assert (
            len(autonomy_turn_end_indices)
            == len(autonomy_turn_start_indices)
            == len(human_turn_end_indices) - 1
            == len(human_turn_start_indices) - 1
        )

        return len(human_turn_start_indices), len(autonomy_turn_start_indices), total_human_turn_time_for_trial

    def compute_mode_switches(self, df):
        autonomy_turn_start_indices, autonomy_turn_end_indices = self.autonomy_turn_markers_per_trial(df)
        human_turn_start_indices, human_turn_end_indices = self.human_turn_markers_per_trial(df)

        human_mode_switches_per_turn_dict = collections.OrderedDict()
        for human_turn_counter, (turn_start_ind, turn_end_ind) in enumerate(
            zip(human_turn_start_indices, human_turn_end_indices)
        ):
            human_mode_switches_for_turn = df["mode_switch_action"][turn_start_ind:turn_end_ind]
            human_mode_switches_for_turn = human_mode_switches_for_turn[human_mode_switches_for_turn.notnull()]
            human_mode_switches_per_turn_dict[human_turn_counter] = len(human_mode_switches_for_turn)

        human_mode_switches_for_trial = sum(human_mode_switches_per_turn_dict.values())

        autonomy_mode_switches_per_turn_dict = collections.OrderedDict()
        for autonomy_turn_counter, (turn_start_ind, turn_end_ind) in enumerate(
            zip(autonomy_turn_start_indices, autonomy_turn_end_indices)
        ):
            autonomy_mode_switches_for_turn = df["mode_switch_action"][turn_start_ind:turn_end_ind]
            autonomy_mode_switches_for_turn = autonomy_mode_switches_for_turn[autonomy_mode_switches_for_turn.notnull()]
            autonomy_mode_switches_per_turn_dict[autonomy_turn_counter] = len(autonomy_mode_switches_for_turn)

        autonomy_mode_switches_for_trial = sum(autonomy_mode_switches_per_turn_dict.values())

        return (
            human_mode_switches_for_trial,
            autonomy_mode_switches_for_trial,
            human_mode_switches_per_turn_dict,
            autonomy_mode_switches_per_turn_dict,
        )

    def compute_trial_time(self, df, f):
        autonomy_turn_start_indices, autonomy_turn_end_indices = self.autonomy_turn_markers_per_trial(df)
        algo_computation_start_indices, algo_computation_end_indices = self.algo_computation_markers_per_trial(df)
        if os.path.split(f)[1] == "l03_disamb_2_16.csv":
            # manually insert a missing 'before'
            algo_computation_start_indices.insert(3, autonomy_turn_start_indices[3] + 2)

        assert (
            len(algo_computation_start_indices)
            == len(algo_computation_end_indices)
            == len(autonomy_turn_start_indices)
            == len(autonomy_turn_end_indices)
        )

        start_index = df["trial_marker"][df["trial_marker"] == "start"].index.tolist()[0]
        end_index = df["trial_marker"][df["trial_marker"] == "end"].index.tolist()[0]

        trial_time_full = df["time"][end_index] - df["time"][start_index]

        computation_time = 0.0
        # accumulate the computation time for entire trial by summing up the computation time for each turn
        for algo_rep, (algo_start_idx, algo_end_idx) in enumerate(
            zip(algo_computation_start_indices, algo_computation_end_indices)
        ):
            computation_time += df["time"][algo_end_idx] - df["time"][algo_start_idx]

        trial_time_minus_computation = trial_time_full - computation_time

        return trial_time_full, trial_time_minus_computation

    def compute_true_goal_probability_gain(self, df):
        autonomy_turn_start_indices, autonomy_turn_end_indices = self.autonomy_turn_markers_per_trial(df)
        pg_red = [eval(v)[0] for v in df["p_g_given_phm"].dropna().values]  # probability associated with red goal

        pg_red_gain = pg_red
        num_time_points = len(pg_red_gain)

        ts = np.linspace(0.0, 1.0, num_time_points)
        base = np.array([1.0] * num_time_points)
        weighted_time = sum(pg_red_gain * ts) / sum(pg_red_gain * base)

        start_time_sec = df["time"][df["p_g_given_phm"].dropna().index[0]]
        end_time_sec = df["time"][df["p_g_given_phm"].dropna().index[-1]]

        times_at_which_autonomy_turn_ends = [df["time"][i] - start_time_sec for i in autonomy_turn_end_indices]
        return pg_red, start_time_sec, end_time_sec, times_at_which_autonomy_turn_ends, weighted_time

    def compute_alpha(self, df):
        human_turn_start_indices, human_turn_end_indices = self.human_turn_markers_per_trial(df)
        human_alpha_assistance_percentage_per_turn_dict = collections.defaultdict(list)
        human_alpha_assistance_to_correct_goal_dict = collections.defaultdict(list)
        human_average_alpha_value_to_correct_goal_per_turn_dict = collections.defaultdict(list)

        # fill in Nans
        df["inferred_goal_str"].fillna(method="ffill", inplace=True)
        df["alpha_val"].fillna(method="ffill", inplace=True)
        for human_turn_counter, (turn_start_ind, turn_end_ind) in enumerate(
            zip(human_turn_start_indices, human_turn_end_indices)
        ):
            alpha_assistance_for_turn = df["alpha_val"][turn_start_ind:turn_end_ind]
            inference_for_turn = df["inferred_goal_str"][turn_start_ind:turn_end_ind]
            assert len(alpha_assistance_for_turn) == len(inference_for_turn)

            inds_for_positive_alpha_for_turn = alpha_assistance_for_turn[alpha_assistance_for_turn > 0.0].index.tolist()
            inds_for_correct_inference_for_turn = inference_for_turn[inference_for_turn == "goal_0"].index.tolist()

            inds_for_positive_alpha_for_correct_inference_for_turn = list(
                set(inds_for_correct_inference_for_turn).intersection(set(inds_for_positive_alpha_for_turn))
            )

            alpha_assistance_positive_to_correct_goal_for_turn = alpha_assistance_for_turn[
                inds_for_positive_alpha_for_correct_inference_for_turn
            ]

            # percentage of time during the turn assistance was present towards correct goal
            alpha_assistance_percentage_per_turn = len(alpha_assistance_positive_to_correct_goal_for_turn) / len(
                alpha_assistance_for_turn
            )
            assert alpha_assistance_percentage_per_turn >= 0.0 and alpha_assistance_percentage_per_turn <= 1.0
            # average alpha value when assistance was provided to correct goal
            if len(alpha_assistance_positive_to_correct_goal_for_turn) > 0:
                average_alpha_value_to_correct_goal_per_turn = np.nanmean(
                    alpha_assistance_positive_to_correct_goal_for_turn.values
                )
            else:
                average_alpha_value_to_correct_goal_per_turn = 0.0

            human_alpha_assistance_percentage_per_turn_dict[human_turn_counter].append(
                alpha_assistance_percentage_per_turn
            )
            human_alpha_assistance_to_correct_goal_dict[human_turn_counter].append(
                alpha_assistance_positive_to_correct_goal_for_turn
            )
            human_average_alpha_value_to_correct_goal_per_turn_dict[human_turn_counter].append(
                average_alpha_value_to_correct_goal_per_turn
            )

        # percetage of time assistance to correct goal was present in the last turn
        assistance_percentage_in_last_turn_to_correct_goal = human_alpha_assistance_percentage_per_turn_dict[
            human_turn_counter
        ][0]
        assistance_val_in_last_turn_to_correct_goal = human_average_alpha_value_to_correct_goal_per_turn_dict[
            human_turn_counter
        ][0]
        # percentage of time (of human turns) assistance was present to the correct goal
        average_assistance_to_correct_goal_for_trial = np.mean(
            list(human_alpha_assistance_percentage_per_turn_dict.values())
        )
        average_alpha_val_to_correct_goal_for_trial = np.mean(
            list(human_average_alpha_value_to_correct_goal_per_turn_dict.values())
        )

        return (
            assistance_percentage_in_last_turn_to_correct_goal,
            assistance_val_in_last_turn_to_correct_goal,
            average_assistance_to_correct_goal_for_trial,
            average_alpha_val_to_correct_goal_for_trial,
            human_alpha_assistance_percentage_per_turn_dict,
            human_average_alpha_value_to_correct_goal_per_turn_dict,
            human_alpha_assistance_to_correct_goal_dict,
        )

    def compute_ratio_of_work(self, df, f):
        autonomy_turn_start_indices, autonomy_turn_end_indices = self.autonomy_turn_markers_per_trial(df)
        human_turn_start_indices, human_turn_end_indices = self.human_turn_markers_per_trial(df)
        algo_computation_start_indices, algo_computation_end_indices = self.algo_computation_markers_per_trial(df)
        if os.path.split(f)[1] == "l03_disamb_2_16.csv":
            # manually insert a missing 'before'
            algo_computation_start_indices.insert(3, autonomy_turn_start_indices[3] + 2)

        assert (
            len(algo_computation_start_indices)
            == len(algo_computation_end_indices)
            == len(autonomy_turn_start_indices)
            == len(autonomy_turn_end_indices)
        )

        # total computation time - to be redcued from autonomy time
        computation_time = 0.0
        # accumulate the computation time for entire trial by summing up the computation time for each turn
        for algo_rep, (algo_start_idx, algo_end_idx) in enumerate(
            zip(algo_computation_start_indices, algo_computation_end_indices)
        ):
            computation_time += df["time"][algo_end_idx] - df["time"][algo_start_idx]

        # compute total human time
        total_human_turn_time_for_trial = 0.0
        for human_turn_counter, (turn_start_ind, turn_end_ind) in enumerate(
            zip(human_turn_start_indices, human_turn_end_indices)
        ):
            total_human_turn_time_for_trial += df["time"][turn_end_ind] - df["time"][turn_start_ind]

        # compute total autonomy time.
        total_autonomy_turn_time_for_trial = 0.0
        for autonomy_turn_counter, (turn_start_ind, turn_end_ind) in enumerate(
            zip(autonomy_turn_start_indices, autonomy_turn_end_indices)
        ):
            total_autonomy_turn_time_for_trial += df["time"][turn_end_ind] - df["time"][turn_start_ind]

        total_autonomy_turn_time_for_trial = total_autonomy_turn_time_for_trial - computation_time
        assert total_autonomy_turn_time_for_trial >= 0.0

        ratio_of_work = total_human_turn_time_for_trial / total_autonomy_turn_time_for_trial
        ratio_of_total_work = total_human_turn_time_for_trial / (
            total_human_turn_time_for_trial + total_autonomy_turn_time_for_trial
        )
        # compute ratio human to autonomy. 1.0, means = equal share. < 1.0 is good, more than one means human had to work more

        return ratio_of_work, ratio_of_total_work

    def compute_metric(self, metric, f, condition_type):
        df = pd.read_csv(f, header=0)
        success = df.inferred_goal_str.dropna().values[-1] == "goal_0"
        print()
        if metric == "mode_switches":
            (
                human_mode_switches_for_trial,
                autonomy_mode_switches_for_trial,
                human_mode_switches_per_turn_dict,
                autonomy_mode_switches_per_turn_dict,
            ) = self.compute_mode_switches(df)
            metric_dict = {
                "human_mode_switches_for_trial": human_mode_switches_for_trial,
                "autonomy_mode_switches_for_trial": autonomy_mode_switches_for_trial,
                "human_mode_switches_per_turn_dict": human_mode_switches_per_turn_dict,
                "autonomy_mode_switches_per_turn_dict": autonomy_mode_switches_per_turn_dict,
            }

        elif metric == "num_turns":
            human_num_turns, autonomy_num_turns, total_human_turn_time_for_trial = self.compute_num_turns(df)
            metric_dict = {
                "human_num_turns": human_num_turns,
                "autonomy_num_turns": autonomy_num_turns,
                "total_human_turn_time_for_trial": total_human_turn_time_for_trial,
            }

        elif metric == "time":
            trial_time_full, trial_time_minus_computation = self.compute_trial_time(df, f)
            metric_dict = {
                "trial_time_full": trial_time_full,
                "trial_time_minus_computation": trial_time_minus_computation,
            }

        elif metric == "true_goal_probability_gain":
            (
                pg_red,
                start_time_sec,
                end_time_sec,
                times_at_which_autonomy_turn_ends,
                weighted_time,
            ) = self.compute_true_goal_probability_gain(df)
            metric_dict = {
                "pg_red": pg_red,
                "start_time_sec": start_time_sec,
                "end_time_sec": end_time_sec,
                "times_at_which_autonomy_turn_ends": times_at_which_autonomy_turn_ends,
                "weighted_time": weighted_time,
            }
        elif metric == "alpha_assistance":
            (
                assistance_percentage_in_last_turn_to_correct_goal,
                assistance_val_in_last_turn_to_correct_goal,
                average_assistance_to_correct_goal_for_trial,
                average_alpha_val_to_correct_goal_for_trial,
                human_alpha_assistance_percentage_per_turn_dict,
                human_average_alpha_value_to_correct_goal_per_turn_dict,
                human_alpha_assistance_to_correct_goal_dict,
            ) = self.compute_alpha(df)
            metric_dict = {
                "assistance_percentage_in_last_turn_to_correct_goal": assistance_percentage_in_last_turn_to_correct_goal,
                "assistance_val_in_last_turn_to_correct_goal": assistance_val_in_last_turn_to_correct_goal,
                "average_assistance_to_correct_goal_for_trial": average_assistance_to_correct_goal_for_trial,
                "average_alpha_val_to_correct_goal_for_trial": average_alpha_val_to_correct_goal_for_trial,
                "human_alpha_assistance_percentage_per_turn_dict": human_alpha_assistance_percentage_per_turn_dict,
                "human_average_alpha_value_to_correct_goal_per_turn_dict": human_average_alpha_value_to_correct_goal_per_turn_dict,
                "human_alpha_assistance_to_correct_goal_dict": human_alpha_assistance_to_correct_goal_dict,
            }
        elif metric == "ratio_of_work":
            ratio_of_work, ratio_of_total_work = self.compute_ratio_of_work(df, f)
            metric_dict = {"ratio_of_work": ratio_of_work, "ratio_of_total_work": ratio_of_total_work}

        return metric_dict, success, df.inferred_goal_str.dropna().values[-1]

    def check_if_file_meets_analysis_condition(self):
        return True

    def group_per_metric(self, metric):

        # if metric == "mode_switches" or metric == "num_turns" or metric == "time":
        metric_dict_all_trials = collections.defaultdict(list)

        for i, f in enumerate(self.files_name_list):  # iterate over each trial

            self.trial_name_info = f.split("_")  # split up string info contained in the file name
            # get the index that contains the word assistance, minus one to get the assistance type
            condition_type = self.trial_name_info[1]  #'control or 'disamb'
            # if not self.trial_name_info[0] in ["l02"]:
            #     continue
            print("filename", f)
            metric_dict, success, last_inferred_goal = self.compute_metric(
                metric, self.files_path_list[i], condition_type
            )
            metric_dict_all_trials[condition_type + "_success"].append(success)
            metric_dict_all_trials[condition_type + "_last_inferred_goal"].append(last_inferred_goal)
            # if not success:
            #     continue

            if metric == "mode_switches":
                metric_dict_all_trials[condition_type + "_human_mode_switches"].append(
                    metric_dict["human_mode_switches_for_trial"]
                )
                metric_dict_all_trials[condition_type + "_turns"].append(
                    metric_dict["human_mode_switches_per_turn_dict"]
                )
            elif metric == "num_turns":
                metric_dict_all_trials[condition_type + "_human_num_turns"].append(metric_dict["human_num_turns"])
                metric_dict_all_trials[condition_type + "_autonomy_num_turns"].append(metric_dict["autonomy_num_turns"])
                metric_dict_all_trials[condition_type + "_total_human_turn_time_for_trial"].append(
                    metric_dict["total_human_turn_time_for_trial"]
                )

            elif metric == "time":
                metric_dict_all_trials[condition_type + "_trial_time_full"].append(metric_dict["trial_time_full"])
                metric_dict_all_trials[condition_type + "_trial_time_minus_computation"].append(
                    metric_dict["trial_time_minus_computation"]
                )
            elif metric == "true_goal_probability_gain":
                metric_dict_all_trials[condition_type + "_pg_red"].append(metric_dict["pg_red"])
                metric_dict_all_trials[condition_type + "_start_time_sec"].append(metric_dict["start_time_sec"])
                metric_dict_all_trials[condition_type + "_end_time_sec"].append(metric_dict["end_time_sec"])
                metric_dict_all_trials[condition_type + "_times_at_which_autonomy_turn_ends"].append(
                    metric_dict["times_at_which_autonomy_turn_ends"]
                )
                metric_dict_all_trials[condition_type + "_weighted_time"].append(metric_dict["weighted_time"])
            elif metric == "alpha_assistance":
                metric_dict_all_trials[condition_type + "_alpha_assistance"].append(
                    metric_dict["average_assistance_to_correct_goal_for_trial"]
                )
                metric_dict_all_trials[condition_type + "_alpha_val"].append(
                    metric_dict["average_alpha_val_to_correct_goal_for_trial"]
                )
                metric_dict_all_trials[condition_type + "_last_turn_alpha_assistance"].append(
                    metric_dict["assistance_percentage_in_last_turn_to_correct_goal"]
                )
                metric_dict_all_trials[condition_type + "_last_turn_alpha_val"].append(
                    metric_dict["assistance_val_in_last_turn_to_correct_goal"]
                )
            elif metric == "ratio_of_work":
                metric_dict_all_trials[condition_type + "_ratio_of_work"].append(metric_dict["ratio_of_work"])
                metric_dict_all_trials[condition_type + "_ratio_of_total_work"].append(
                    metric_dict["ratio_of_total_work"]
                )
        return metric_dict_all_trials

    def create_dataframe(self, data, metric):
        # assumes labels are in the same order of the data
        # assign the assistance condition label to each data in the arrays so we can add it as a column in the dataframe
        condition = []
        for i in range(len(self.labels)):
            for j in range(len(data[i])):
                condition.append(self.labels[i])
        df = pd.DataFrame(
            {metric: list(itertools.chain(*data)), "condition": condition}
        )  # flatten the data and create dataframe so each value corresponds with assistance condition
        return df

    def data_analysis(self):
        for metric in ["num_turns"]:
            metric_dict_all_trials = self.group_per_metric(metric)
            print(
                "SUCCESS CONTROL DISMAB",
                sum(metric_dict_all_trials["control_success"]) / len(metric_dict_all_trials["control_success"]),
                sum(metric_dict_all_trials["disamb_success"]) / len(metric_dict_all_trials["disamb_success"]),
            )

            if metric == "mode_switches":
                metric_dict_all_trials["control_human_mode_switches"] = [
                    e for e in metric_dict_all_trials["control_human_mode_switches"] if e <= 10
                ]
                metric_dict_all_trials["disamb_human_mode_switches"] = [
                    e for e in metric_dict_all_trials["disamb_human_mode_switches"] if e <= 10
                ]
                data = [
                    metric_dict_all_trials["control_human_mode_switches"],
                    metric_dict_all_trials["disamb_human_mode_switches"],
                ]
                print("MODE SWITCHES - CONTROl, DISAMB", np.mean(data[0]), np.mean(data[1]))
                self.parametric_anova_with_post_hoc(data, metric)
            elif metric == "num_turns":
                metric_dict_all_trials["control_human_num_turns"] = [
                    e for e in metric_dict_all_trials["control_human_num_turns"] if e <= 10
                ]
                metric_dict_all_trials["disamb_human_num_turns"] = [
                    e for e in metric_dict_all_trials["disamb_human_num_turns"] if e <= 10
                ]
                data = [
                    metric_dict_all_trials["control_human_num_turns"],
                    metric_dict_all_trials["disamb_human_num_turns"],
                ]
                print(
                    "NUM TURNS - CONTROl, DISAMB", np.mean(data[0]), np.std(data[0]), np.mean(data[1]), np.std(data[1])
                )
                self.parametric_anova_with_post_hoc(data, "num_turns")
                data = [
                    metric_dict_all_trials["control_total_human_turn_time_for_trial"],
                    metric_dict_all_trials["disamb_total_human_turn_time_for_trial"],
                ]
                print("HUMAN OPERATING TIME - CONTROl, DISAMB", np.mean(data[0]), np.mean(data[1]))
                self.parametric_anova_with_post_hoc(data, "human_operating_time")
            elif metric == "time":
                metric_dict_all_trials["control_trial_time_minus_computation"] = [
                    e for e in metric_dict_all_trials["control_trial_time_minus_computation"] if e <= 100
                ]
                metric_dict_all_trials["disamb_trial_time_minus_computation"] = [
                    e for e in metric_dict_all_trials["disamb_trial_time_minus_computation"] if e <= 100
                ]
                data = [
                    metric_dict_all_trials["control_trial_time_minus_computation"],
                    metric_dict_all_trials["disamb_trial_time_minus_computation"],
                ]
                print("TIME - CONTROl, DISAMB", np.mean(data[0]), np.mean(data[1]))
                self.parametric_anova_with_post_hoc(data, metric)
            elif metric == "true_goal_probability_gain":
                # self.plot_true_goal_probability_gains(
                #     metric_dict_all_trials["control_pg_red"],
                #     metric_dict_all_trials["control_start_time_sec"],
                #     metric_dict_all_trials["control_end_time_sec"],
                #     metric_dict_all_trials["control_times_at_which_autonomy_turn_ends"],
                #     "control",
                # )

                # self.plot_true_goal_probability_gains(
                #     metric_dict_all_trials["disamb_pg_red"],
                #     metric_dict_all_trials["disamb_start_time_sec"],
                #     metric_dict_all_trials["disamb_end_time_sec"],
                #     metric_dict_all_trials["disamb_times_at_which_autonomy_turn_ends"],
                #     "disamb",
                # )
                data = [
                    metric_dict_all_trials["control_weighted_time"],
                    metric_dict_all_trials["disamb_weighted_time"],
                ]
                print("NUM TURNS - CONTROl, DISAMB", np.mean(data[0]), np.mean(data[1]))
            elif metric == "alpha_assistance":

                data = [
                    100.0 * np.array(metric_dict_all_trials["control_alpha_assistance"]),
                    100.0 * np.array(metric_dict_all_trials["disamb_alpha_assistance"]),
                ]
                print("ASSISTANCE PERCENTAGE - CONTROl, DISAMB", np.mean(data[0]), np.mean(data[1]))
                self.parametric_anova_with_post_hoc(data, metric)
                data = [
                    metric_dict_all_trials["control_alpha_val"],
                    metric_dict_all_trials["disamb_alpha_val"],
                ]
                print("ASSISTANCE STRENGTH - CONTROl, DISAMB", np.mean(data[0]), np.mean(data[1]))
                self.parametric_anova_with_post_hoc(data, "alpha_val")
                # data = [
                #     metric_dict_all_trials["control_last_turn_alpha_assistance"],
                #     metric_dict_all_trials["disamb_last_turn_alpha_assistance"],
                # ]
                # data = [
                #     metric_dict_all_trials["control_last_turn_alpha_val"],
                #     metric_dict_all_trials["disamb_last_turn_alpha_val"],
                # ]
            elif metric == "ratio_of_work":
                data = [
                    np.array(metric_dict_all_trials["control_ratio_of_total_work"]),
                    np.array(metric_dict_all_trials["disamb_ratio_of_total_work"]),
                ]
                print("RATIO OF WORK - CONTROl, DISAMB", np.mean(data[0]), np.mean(data[1]))
                self.parametric_anova_with_post_hoc(data, metric)

    def plot_true_goal_probability_gains(
        self, pg_red_list, start_time_sec_list, end_time_sec_list, times_at_which_autonomy_turn_ends_list, condition
    ):

        for pg_red, start_time, end_time, times_at_which_autonomy_turn_ends in zip(
            pg_red_list, start_time_sec_list, end_time_sec_list, times_at_which_autonomy_turn_ends_list
        ):
            num_points = len(pg_red)
            x_list = np.linspace(0.0, end_time - start_time, num_points - 1)
            plt.plot(x_list, np.diff(pg_red), color=COLOR_LIST[condition])

        plt.show()

    def plot_with_significance(self, df, metric, *args, **kwargs):

        font_size_dict = {
            "time": 18,
            "alpha_val": 18,
            "alpha_assistance": 18,
            "human_operating_time": 18,
            "num_turns": 18,
            "mode_switches": 18,
            "ratio_of_work": 18,
        }
        p_star_delta_dict = {
            "time": 0.1,
            "alpha_val": 0.085,
            "alpha_assistance": 0.1,
            "human_operating_time": 0.01,
            "num_turns": 0.1,
            "mode_switches": 0.08,
            "ratio_of_work": 0.01,
        }
        text_offset = {
            "time": 0.01,
            "alpha_val": 0.01,
            "alpha_assistance": 0.01,
            "human_operating_time": 0.01,
            "num_turns": 0.01,
            "mode_switches": 0.01,
            "ratio_of_work": 0.005,
        }

        pairs = kwargs.get("pairs", None)
        p_values = kwargs.get("p_values", None)

        sns.set_style("dark")
        sns.set_context("paper")
        sns.set_palette("colorblind")

        # ax = sns.barplot(x=df["condition"], y=df[metric], data=df)
        # import IPython

        # IPython.embed(banner1="check")

        # ax = sns.boxplot(x=df["condition"], y=df[metric])
        # ax = sns.swarmplot(x=df["condition"], y=df[metric], color=".4")
        ax = sns.violinplot(x=df["condition"], y=df[metric])
        font_size = font_size_dict[metric]
        ax.tick_params(labelsize=font_size)

        plt.subplots_adjust(left=0.17)
        plt.subplots_adjust(right=0.96)

        # If significance exists, plot it
        if not pairs == None:
            y_min = df[metric].max()  # get maximum data value (max y)
            h = y_min * p_star_delta_dict[metric]
            y_min = y_min + h

            sig_text = []
            for i in p_values:
                if i <= self.v_strong_alpha:
                    sig_text.append("***")
                elif i <= self.strong_alpha:
                    sig_text.append("**")
                elif i <= self.alpha:
                    sig_text.append("*")

            # get y position for all the p-value pairs based on location and significacne
            sig_df = pd.DataFrame({"pair": pairs, "p_value": p_values, "text": sig_text})
            # sort so it looks neat when plotting. convert to data frame so can get sig_text with the pairs after sorting
            sig_df = sig_df.sort_values(by=["pair"])
            sig_df.reset_index(drop=True, inplace=True)  # reindex after sorting

            for i in range(len(pairs)):
                y_pos = [y_min + (h * i)] * 2  # start and end is same height so *2
                # text position should be in the center of the line connecting the pairs
                text_pos_x = sum(sig_df.loc[i, "pair"]) / 2
                text_pos_y = y_min + (h * i) + text_offset[metric]
                plt.plot(sig_df.loc[i, "pair"], y_pos, lw=2, c="k")
                plt.text(
                    text_pos_x,
                    text_pos_y,
                    sig_df.loc[i, "text"],
                    ha="center",
                    va="bottom",
                    color="k",
                    fontsize=font_size - 2,
                )

        plt.xlabel("")

        # To do: Clean lablels:
        if metric == "time":
            plt.ylabel("Total Trial Time (s)", fontsize=font_size)
        elif metric == "mode_switches":
            plt.ylabel("Number of Mode Switches Per Trial", fontsize=font_size)
            plt.ylim(0, 11)
        elif metric == "corrections":
            plt.ylabel("Average Intervention Counts", fontsize=font_size)
        elif metric == "success":
            plt.ylabel("Percentage of Successful Trials (%)", fontsize=font_size)
        elif metric == "human_operating_time":
            plt.ylabel("Operating Time Per Trial for Humans (s)", fontsize=font_size)
        elif metric == "num_turns":
            plt.ylabel("Number of Human Turns Per Trial", fontsize=font_size)
            plt.ylim(0.0, 12.0)
        elif metric == "alpha_val":
            plt.ylabel("Strength of Autonomy Assistance", fontsize=font_size)
            plt.ylim(0.0, 0.9)
        elif metric == "alpha_assistance":
            plt.ylabel("Autonomy Assistance Engagement (%)", fontsize=font_size)
            plt.ylim(0.0, 85.0)
        elif metric == "ratio_of_work":
            plt.ylabel("Normalized Human Operating Time Per Trial", fontsize=font_size)
            plt.ylim(0.6, 1.1)

        plt.show()

    def get_significant_pairs(self, df, metric):

        pairwise_comparisons = sp.posthoc_conover(df, val_col=metric, group_col="condition", p_adjust="holm")
        # embed()
        # TO DO: Wilcoxon won't work for mode switches because not truly paired test (conditions have different lengths)
        # pairwise_comparisons = sp.posthoc_wilcoxon(df, val_col=metric, group_col='condition', p_adjust='holm')

        groups = pairwise_comparisons.keys().to_list()
        combinations = list(itertools.combinations(groups, 2))  # possible combinations for pairwise comparison
        pairs = []
        p_values = []
        # get pairs for x:
        for i in range(len(combinations)):
            # if signifcane between the two pairs is alot, add position
            if pairwise_comparisons.loc[combinations[i][0], combinations[i][1]] <= self.alpha:
                pairs.append([self.label_to_plot_pos[combinations[i][0]], self.label_to_plot_pos[combinations[i][1]]])
                p_values.append(pairwise_comparisons.loc[combinations[i][0], combinations[i][1]])

        return pairs, p_values

    def parametric_anova_with_post_hoc(self, data, metric):

        df = self.create_dataframe(data, metric)

        # non parametric kruskal wallis test
        H, p = ss.kruskal(*data)
        # if can reject null hypothesis that population medians of all groups are equel,
        if p <= self.alpha:
            print("P_VALUE", p)
            # do posthoc test to learn which groups differ in their medians
            pairs, p_values = self.get_significant_pairs(df, metric)
            self.plot_with_significance(df, metric, pairs=pairs, p_values=p_values)

        else:
            print(metric + " failed hypothesis test.")
            self.plot_with_significance(df, metric)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-id", "--subject_id", help="experiment block: subject_id_type_assistance_block", type=str)
    parser.add_argument(
        "-m",
        "--metrics",
        help="metrics to analyze",
        nargs="+",
        default=["time", "mode_switches", "turns", "distance_covered", "information_gain"],
    )
    parser.add_argument(
        "-p",
        "--directory_path",
        help="path to folder which contains both trial_csvs and trial_dir folders",
        default=os.path.join("N:", "2020-IROS-UnintendedCommandAssistance", "data"),
    )
    args = parser.parse_args()
    if args.subject_id:
        comp_assistance = CompareAssistanceParadigms(args)

    else:
        comp_assistance = CompareAssistanceParadigms(args)
    comp_assistance.load_trial_data()
    comp_assistance.data_analysis()
