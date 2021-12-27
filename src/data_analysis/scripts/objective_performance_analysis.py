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
from analysis_base_class import AnalysisBase
import scikit_posthocs as sp
import collections

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


class CompareAssistanceParadigms(object):
    def __init__(self, args):

        self.directory_list = list()
        self.directory_path = args.directory_path
        self.metrics = args.metrics

        self.data_dir = os.path.join(self.directory_path, "parsed_trial_data")
        print(self.data_dir)
        self.get_recursive_folders()
        self.labels = ["Control", "Disamb"]
        self.assistance_cond = ["control", "disamb"]
        self.label_to_plot_pos = {"Control": 0, "Disamb": 1}
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

    def compute_num_turns(self, df):
        autonomy_turn_start_indices, autonomy_turn_end_indices = self.autonomy_turn_markers_per_trial(df)
        human_turn_start_indices, human_turn_end_indices = self.human_turn_markers_per_trial(df)

        assert (
            len(autonomy_turn_end_indices)
            == len(autonomy_turn_start_indices)
            == len(human_turn_end_indices) - 1
            == len(human_turn_start_indices) - 1
        )

        return len(human_turn_start_indices), len(autonomy_turn_start_indices)

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

    def compute_metric(self, metric, f, condition_type):
        df = pd.read_csv(f, header=0)

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
            human_num_turns, autonomy_num_turns = self.compute_num_turns(df)
            metric_dict = {"human_num_turns": human_num_turns, "autonomy_num_turns": autonomy_num_turns}

        return metric_dict

    def group_per_metric(self, metric):

        if metric == "mode_switches" or metric == "num_turns":
            metric_dict_all_trials = collections.defaultdict(list)

        for i, f in enumerate(self.files_name_list):  # iterate over each trial
            print("filename", f)
            self.trial_name_info = f.split("_")  # split up string info contained in the file name
            # get the index that contains the word assistance, minus one to get the assistance type
            condition_type = self.trial_name_info[1]  #'control or 'disamb'
            metric_dict = self.compute_metric(metric, self.files_path_list[i], condition_type)

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

        if metric == "mode_switches" or metric == "num_turns":
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
        for metric in ["mode_switches", "num_turns"]:
            metric_dict_all_trials = self.group_per_metric(metric)
            if metric == "mode_switches":
                data = [
                    metric_dict_all_trials["control_human_mode_switches"],
                    metric_dict_all_trials["disamb_human_mode_switches"],
                ]
                self.parametric_anova_with_post_hoc(data, metric)
            elif metric == "num_turns":
                data = [
                    metric_dict_all_trials["control_human_num_turns"],
                    metric_dict_all_trials["disamb_human_num_turns"],
                ]
                print(np.mean(data[0]), np.mean(data[1]))
                self.parametric_anova_with_post_hoc(data, metric)

    def plot_with_significance(self, df, metric, *args, **kwargs):

        pairs = kwargs.get("pairs", None)
        p_values = kwargs.get("p_values", None)

        sns.set_style("dark")
        sns.set_context("paper")
        sns.set_palette("colorblind")

        # ax = sns.barplot(x=df["condition"], y=df[metric], data=df)

        ax = sns.boxplot(x=df["condition"], y=df[metric])
        ax = sns.swarmplot(x=df["condition"], y=df[metric], color=".4")
        font_size = 20
        ax.tick_params(labelsize=font_size)

        plt.subplots_adjust(left=0.17)
        plt.subplots_adjust(right=0.96)

        # If significance exists, plot it
        if not pairs == None:
            y_min = round(df[metric].max())  # get maximum data value (max y)
            h = y_min * 0.1
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
            sig_df = sig_df.sort_values(
                by=["pair"]
            )  # sort so it looks neat when plotting. convert to data frame so can get sig_text with the pairs after sorting
            sig_df.reset_index(drop=True, inplace=True)  # reindex after sorting

            for i in range(len(pairs)):
                y_pos = [y_min + (h * i)] * 2  # start and end is same height so *2
                text_pos_x = (
                    sum(sig_df.loc[i, "pair"]) / 2
                )  # text position should be in the center of the line connecting the pairs
                text_pos_y = y_min + (h * i) + 0.25
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
            plt.ylabel("Successful Trial Mode Switch Count", fontsize=font_size)
            plt.ylim(0, 30)
        elif metric == "corrections":
            plt.ylabel("Average Intervention Counts", fontsize=font_size)
        elif metric == "success":
            plt.ylabel("Percentage of Successful Trials (%)", fontsize=font_size)
        elif metric == "distance":
            plt.ylabel("Distance to Goal", fontsize=font_size)

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
            if (
                pairwise_comparisons.loc[combinations[i][0], combinations[i][1]] <= self.alpha
            ):  # if signifcane between the two pairs is alot, add position
                pairs.append([self.label_to_plot_pos[combinations[i][0]], self.label_to_plot_pos[combinations[i][1]]])
                p_values.append(pairwise_comparisons.loc[combinations[i][0], combinations[i][1]])

        return pairs, p_values

    def parametric_anova_with_post_hoc(self, data, metric):

        df = self.create_dataframe(data, metric)

        # non parametric kruskal wallis test
        H, p = ss.kruskal(*data)
        # if can reject null hypothesis that population medians of all groups are equel,
        if p <= self.alpha:
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
