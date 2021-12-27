#!/usr/bin/env python3
# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

import os
import argparse
import collections
import pandas as pd
import numpy as np
import pickle
from ast import literal_eval


# read in topic csvs of interest with name and index
# remove unused columns from topic csv
# ensure time is ascending
# merge/concatenate all with single rosbagTimeStamp column
# divide up into trial dataframes
# save as csv with corresponding pkl file name (subject_#_assistance_block_#_trial_#.csv)


class ConcatenateMainStudyTopicsPerTrial(object):
    def __init__(self, args):
        # this is where the unified dict will eventually be saved.
        self.data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data_processing", "raw_data"
        )
        self.data_analysis_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data_analysis",
            "parsed_trial_data",
        )
        try:
            os.makedirs(self.data_analysis_dir)
        except:
            pass

        # all folders in raw_data dir. subject_block_id format. Each such folder has different csv's corresponding to each of the topics recorded during the session
        self.all_subject_block_dirs = os.listdir(self.data_dir)
        # remove cetrain folders.
        self.ignore_list = ["andrew", "deepak", "j01", "j02", "l01", "given"]

        def should_ignore(folder_name):
            return any(ignore_cond in folder_name for ignore_cond in self.ignore_list)

        self.all_subject_block_dirs = filter(lambda n: not should_ignore(n), self.all_subject_block_dirs)
        self.all_subject_block_dirs = sorted(list(self.all_subject_block_dirs))

        # To Do: make these inputs to function
        self.topics = {
            "alpha": "_alpha.csv",
            "argmax_goals": "_argmax_goals.csv",
            "autonomy_turn_target": "_autonomy_turn_target.csv",
            "autonomy_vel": "_autonomy_vel.csv",
            "belief_info": "_belief_info.csv",
            "blend_vel": "_blend_vel.csv",
            "freeze_update": "_freeze_update.csv",
            "function_timer": "_function_timer.csv",
            "has_human_initiated": "_has_human_initiated.csv",
            "human_vel": "_human_vel.csv",
            "inferred_goal": "_inferred_goal.csv",
            # "joy_sip_puff_before": "_joy_sip_puff_before.csv",
            # "joy_sip_puff": "_joy_sip_puff.csv",
            # "joy": "_joy.csv",
            "mode_switch": "_mode_switch.csv",
            "robot_state": "_robot_state.csv",
            "shutdown": "_shutdown.csv",
            "trial_index": "_trial_index.csv",
            "trial_marker": "_trial_marker.csv",
            "turn_indicator": "_turn_indicator.csv",
            "user_vel": "_user_vel.csv",
        }

        self.sub_topics = {
            "alpha": ["rosbagTimestamp", "data"],
            "argmax_goals": ["rosbagTimestamp", "data"],
            "autonomy_turn_target": ["rosbagTimestamp", "robot_continuous_position", "robot_continuous_orientation"],
            "autonomy_vel": ["rosbagTimestamp", "data"],
            "belief_info": ["rosbagTimestamp", "p_g_given_phm"],
            "blend_vel": ["rosbagTimestamp", "data"],
            "freeze_update": ["rosbagTimestamp", "data"],
            "function_timer": ["rosbagTimestamp", "data"],
            "has_human_initiated": ["rosbagTimestamp", "data"],
            "human_vel": ["rosbagTimestamp", "data"],
            "inferred_goal": ["rosbagTimestamp", "data"],
            # "joy_sip_puff_before": ["rosbagTimestamp", "frame_id", "axes", "buttons"],
            # "joy_sip_puff": ["rosbagTimestamp", "frame_id", "axes", "buttons"],
            # "joy": ["rosbagTimestamp", "frame_id", "axes", "buttons"],
            "mode_switch": ["rosbagTimestamp", "data"],
            "robot_state": [
                "rosbagTimestamp",
                "robot_continuous_position",
                "robot_continuous_orientation",
                "robot_linear_velocity",
                "robot_angular_velocity",
                "robot_discrete_state",
                "discrete_x",
                "discrete_y",
                "discrete_orientation",
                "discrete_mode",
            ],
            "shutdown": ["rosbagTimestamp", "data"],
            "trial_index": ["rosbagTimestamp", "data"],
            "trial_marker": ["rosbagTimestamp", "data"],
            "turn_indicator": ["rosbagTimestamp", "data"],
            "user_vel": ["rosbagTimestamp", "interface_signal", "mode_switch"],
        }

        self.total_num_valid_trials_for_all_subjects = 0
        self.max_trials_per_block = 8

        self.skipped_trials = {}

    def ensure_ascending(self, data):
        if sorted(list(data)) == list(data):
            return True
        else:
            return False

    def ensure_all_trials_marked(self, df_dict_subject_block):
        num_trials_per_block = len(df_dict_subject_block["trial_index"])
        start_markers = df_dict_subject_block["trial_marker"].loc[
            df_dict_subject_block["trial_marker"]["data"] == "start"
        ]
        # end_markers = df_dict_subject_block['trial_marker'].loc[df_dict_subject_block['trial_marker']['data']=='end']
        restart_markers = df_dict_subject_block["trial_marker"].loc[
            df_dict_subject_block["trial_marker"]["data"] == "skipped"
        ]

        if len(start_markers) != num_trials_per_block:
            # number of starts + restarts should == number of times trial index was published
            if not len(restart_markers) == (num_trials_per_block - len(start_markers)):
                return False

        return True, num_trials_per_block

    def rename_df_column(self, df_dict_subject_assistance_block):
        df_dict_subject_assistance_block["alpha"].rename(columns={"data": "alpha_val"}, inplace=True)
        if "argmax_goals" in df_dict_subject_assistance_block.keys():
            df_dict_subject_assistance_block["argmax_goals"].rename(columns={"data": "argmax_goals"}, inplace=True)
        df_dict_subject_assistance_block["autonomy_vel"].rename(columns={"data": "autonomy_vel"}, inplace=True)
        df_dict_subject_assistance_block["blend_vel"].rename(columns={"data": "blend_vel"}, inplace=True)
        df_dict_subject_assistance_block["human_vel"].rename(columns={"data": "human_vel"}, inplace=True)

        df_dict_subject_assistance_block["freeze_update"].rename(
            columns={"data": "belief_update_freeze_marker"}, inplace=True
        )
        df_dict_subject_assistance_block["function_timer"].rename(
            columns={"data": "assistance_compute_marker"}, inplace=True
        )
        df_dict_subject_assistance_block["has_human_initiated"].rename(
            columns={"data": "human_initiated_marker"}, inplace=True
        )
        df_dict_subject_assistance_block["inferred_goal"].rename(columns={"data": "inferred_goal_str"}, inplace=True)
        df_dict_subject_assistance_block["mode_switch"].rename(columns={"data": "mode_switch_action"}, inplace=True)
        df_dict_subject_assistance_block["shutdown"].rename(columns={"data": "shutdown_message"}, inplace=True)
        df_dict_subject_assistance_block["trial_marker"].rename(columns={"data": "trial_marker"}, inplace=True)
        df_dict_subject_assistance_block["trial_index"].rename(columns={"data": "trial_index"}, inplace=True)
        df_dict_subject_assistance_block["turn_indicator"].rename(columns={"data": "turn_indicator"}, inplace=True)

        for df in df_dict_subject_assistance_block.keys():  # replace the timestamp column name
            df_dict_subject_assistance_block[df].rename(columns={"rosbagTimestamp": "time"}, inplace=True)

        return df_dict_subject_assistance_block

    def get_trial_indices(self, df_dict_subject_block):
        # assumes already changed name of column
        bool_start = False
        start_ind = []
        end_ind = []
        for i in range(len(df_dict_subject_block["trial_marker"])):
            # get start index
            if df_dict_subject_block["trial_marker"].loc[i, "trial_marker"] == "start" and bool_start == False:
                start_ind.append(df_dict_subject_block["trial_marker"].loc[i, "time"])
                bool_start = True
            # get end index
            elif df_dict_subject_block["trial_marker"].loc[i, "trial_marker"] == "end" and bool_start == True:
                end_ind.append(df_dict_subject_block["trial_marker"].loc[i, "time"])
                bool_start = False
            # if was reset, reset start index
            elif df_dict_subject_block["trial_marker"].loc[i, "trial_marker"] == "trial_skipped" and bool_start == True:
                start_ind.pop()  # pop out last element from start, not true start
                start_ind.append(None)
                end_ind.append(None)
                bool_start = False
            else:
                print("end marker after trial_skipped marker")

        assert len(start_ind) == len(end_ind)
        return start_ind, end_ind

    def parse_testing_data(self):
        # takes data from all subjects and all blocks and puts them into a single pickle file.
        overall_trial_index = 0
        # iterate through all the subject and block directory. each of these directories contain separate csvs for each of the topics recorded in the bag file extracted using the shell script
        for subject_block_dir in self.all_subject_block_dirs:
            full_path_subject_block_dir = os.path.join(self.data_dir, subject_block_dir)
            subject_id = subject_block_dir.split("_")[0]
            assistance_condition = subject_block_dir.split("_")[1]
            block_id = subject_block_dir.split("_")[-1]
            print(full_path_subject_block_dir, subject_id, assistance_condition, block_id)

            df_dict_subject_assistance_block = collections.OrderedDict()
            for topic, topic_csv in self.topics.items():
                csv_file_for_topic_for_subject_assistance_block = os.path.join(full_path_subject_block_dir, topic_csv)
                if os.path.exists(csv_file_for_topic_for_subject_assistance_block):
                    # print("Loading csv for topic", topic)
                    df = pd.read_csv(
                        csv_file_for_topic_for_subject_assistance_block, header=0, usecols=self.sub_topics[topic]
                    )
                    keys = [key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ["object"]]
                    if topic == "argmax_goals":
                        continue
                    for key in keys:
                        df[key].apply(lambda s: s.replace('"', ""))  # remove double quotation
                        # convert data types in [] where some rows are zero (considers mixed so dtype is object instead of int or float array)
                        if topic == "argmax_goals" and key == "data":
                            continue
                        # print(topic, key)
                        df[key] = df[key].apply(literal_eval)

                    # sanity check, ensure rosbag times in ascending order
                    assert self.ensure_ascending(df.rosbagTimestamp)
                    df_dict_subject_assistance_block[topic] = df

            if "autonomy_turn_target" in df_dict_subject_assistance_block.keys():
                # need to do this renaming to avoid column name clashes during merging
                df_dict_subject_assistance_block["autonomy_turn_target"] = df_dict_subject_assistance_block[
                    "autonomy_turn_target"
                ].rename(
                    columns={
                        "robot_continuous_position": "att_target_robot_continuous_position",
                        "robot_continuous_orientation": "att_target_robot_continuous_orientation",
                    }
                )
            is_marked, num_trials_per_block = self.ensure_all_trials_marked(df_dict_subject_assistance_block)
            assert is_marked  # makes sure that the all trials had the 'start marker'

            df_dict_subject_assistance_block = self.rename_df_column(df_dict_subject_assistance_block)
            # start and end indices of each trial in a block.

            trial_start_indices_subject_block, trial_end_indices_subject_block = self.get_trial_indices(
                df_dict_subject_assistance_block
            )
            assert len(trial_start_indices_subject_block) == len(trial_end_indices_subject_block)

            topics = df_dict_subject_assistance_block.keys()
            # list of pandas frames
            topic_frames = [df_dict_subject_assistance_block[topic] for topic in topics]
            combined_data_frame_subject_block = topic_frames[0]

            for i in range(0, len(topic_frames)):
                combined_data_frame_subject_block = combined_data_frame_subject_block.merge(
                    topic_frames[i], how="outer", sort=True
                )

            num_valid_trials_per_block = len(trial_start_indices_subject_block)
            assert num_valid_trials_per_block == self.max_trials_per_block  # might have Nones

            # those trials which were never skipped by pressing P, but marked as skipped due to error in data collection procedure.
            if subject_block_dir in self.skipped_trials.keys():
                trial_inds_to_be_skipped = self.skipped_trials[subject_block_dir]
                for trial_ind in trial_inds_to_be_skipped:
                    trial_start_indices_subject_block.pop(trial_ind)
                    trial_end_indices_subject_block.pop(trial_ind)

            assert len(trial_start_indices_subject_block) == len(trial_end_indices_subject_block)

            # Remove any Nones. These were created by pressing P during a trial. Forced skip.
            if None in trial_start_indices_subject_block:
                trial_start_indices_subject_block = [
                    trial_ind for trial_ind in trial_start_indices_subject_block if trial_ind is not None
                ]
                trial_end_indices_subject_block = [
                    trial_ind for trial_ind in trial_end_indices_subject_block if trial_ind is not None
                ]

            assert len(trial_start_indices_subject_block) == len(trial_end_indices_subject_block)

            num_valid_trials_per_block = len(trial_start_indices_subject_block)
            self.total_num_valid_trials_for_all_subjects += num_valid_trials_per_block

            # path to store trial data for subject:
            trial_dir = os.path.join(self.data_analysis_dir, subject_id)
            try:
                os.makedirs(trial_dir)
            except:
                pass

            for i in range(num_valid_trials_per_block):
                print("Valid trial num within block", i)
                # extract the data frame (all topics) for the ith trial.
                trial_data_frame = combined_data_frame_subject_block.loc[
                    (combined_data_frame_subject_block["time"] >= trial_start_indices_subject_block[i] - 0.1)
                    & (combined_data_frame_subject_block["time"] <= trial_end_indices_subject_block[i])
                ]
                # reset index for the dataframe so starts from 0
                trial_data_frame.reset_index(drop=True, inplace=True)
                ts = trial_data_frame.loc[0, "time"]  # first time stamp of the trial
                for j in range(len(trial_data_frame)):
                    # Adjust the timestamps. start from 0.0s for each trial
                    trial_data_frame.at[j, "time"] = trial_data_frame.loc[j, "time"] - ts

                # grab trial specific information
                try:
                    # make sure the index was only logged ONCE during the trial
                    assert np.sum(~np.isnan(trial_data_frame["trial_index"].values))
                except:
                    import IPython

                    IPython.embed(banner1="catch")

                # What was the original trial index for this particular trial.
                trial_index_for_trial = int(
                    trial_data_frame[~np.isnan(trial_data_frame["trial_index"].values)]["trial_index"].values[0]
                )

                trial_filename = os.path.join(
                    trial_dir,
                    subject_id
                    + "_"
                    + assistance_condition
                    + "_"
                    + block_id
                    + "_"
                    + str(trial_index_for_trial)
                    + ".csv",
                )

                trial_data_frame.to_csv(trial_filename, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--block", help="experiment block: subject_id_type_assistance_block", type=str)
    args = parser.parse_args()
    block_to_trial = ConcatenateMainStudyTopicsPerTrial(args)
    block_to_trial.parse_testing_data()

# python main_study_concatenate_topics_per_trial -block l01_control_condition_0
