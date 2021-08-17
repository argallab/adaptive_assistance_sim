#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

import sys
from corrective_mode_switch_utils import SCALE, VIEWPORT_W, VIEWPORT_H, ROBOT_RADIUS, GOAL_RADIUS, PI
from corrective_mode_switch_utils import AssistanceType, StartDirection, RGOrient
import numpy as np
import os
import pickle
import collections
import itertools
import argparse
import random

VIEWPORT_WS = VIEWPORT_W/SCALE
VIEWPORT_HS = VIEWPORT_H/SCALE
ROBOT_RADIUS_S = ROBOT_RADIUS/SCALE
GOAL_RADIUS_S = GOAL_RADIUS/SCALE

# RG_CONFIGS = [RGOrient.TOP_RIGHT, RGOrient.TOP_LEFT, RGOrient.BOTTOM_LEFT, RGOrient.BOTTOM_RIGHT]
RG_CONFIGS = [RGOrient.TOP_LEFT, RGOrient.BOTTOM_RIGHT]
NUM_TURNS = [3]
START_DIRECTIONS = [StartDirection.X, StartDirection.Y]
ASSISTANCE_TYPES = [0, 1, 2]

ROBOT_GOAL_CONFIGURATIONS = {RGOrient.TOP_RIGHT:{'robot':[VIEWPORT_WS/4, VIEWPORT_HS/4],     'goal':[3*VIEWPORT_WS/4, 3*VIEWPORT_HS/4]},
                             RGOrient.TOP_LEFT:{'robot':[3*VIEWPORT_WS/4, VIEWPORT_HS/4],   'goal':[VIEWPORT_WS/4, 3*VIEWPORT_HS/4]},
                             RGOrient.BOTTOM_LEFT:{'robot':[3*VIEWPORT_WS/4, 3*VIEWPORT_HS/4], 'goal':[VIEWPORT_WS/4, VIEWPORT_HS/4]},
                             RGOrient.BOTTOM_RIGHT:{'robot':[VIEWPORT_WS/4, 3*VIEWPORT_HS/4],   'goal':[3*VIEWPORT_WS/4, VIEWPORT_HS/4]}
                            }
START_MODE_DICT = {StartDirection.X: {-1: 't', 1: 'y'},
                   StartDirection.Y: {-1: 'x', 1: 't'}}

START_MODE_OPTIONS = [-1, 1] # if (x,y, theta) is the mode sequence, -1 refers to the mode to the 'left' of the start direction and +1 refers to the mode on the right direction.

def generate_experiment_trials(args):
    num_reps_per_condition = args.num_reps_per_condition
    trial_dir = args.trial_dir
    metadata_dir = args.metadata_dir
    if not os.path.exists(trial_dir):
        os.makedirs(trial_dir)

    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)

    index = 0
    assistance_to_pkl_index = collections.defaultdict(list)

    for rg_config, num_turns, start_direction, assistance_type  in itertools.product(RG_CONFIGS, NUM_TURNS, START_DIRECTIONS, ASSISTANCE_TYPES):
        trial_info_dict = collections.OrderedDict()
        trial_info_dict['env_params'] = collections.OrderedDict()
        trial_info_dict['env_params']['r_to_g_relative_orientation'] = rg_config
        trial_info_dict['env_params']['num_turns'] = num_turns
        trial_info_dict['env_params']['start_direction'] = start_direction
        trial_info_dict['env_params']['assistance_type'] = assistance_type

        trial_info_dict['env_params']['robot_position'] = ROBOT_GOAL_CONFIGURATIONS[rg_config]['robot']
        trial_info_dict['env_params']['robot_orientation'] = 0.0
        trial_info_dict['env_params']['goal_position'] = ROBOT_GOAL_CONFIGURATIONS[rg_config]['goal']
        if index % 2 == 0:
            trial_info_dict['env_params']['goal_orientation'] = PI/2 #randomly pick PI/2 or -PI/2
        else:
            trial_info_dict['env_params']['goal_orientation'] = -PI/2

        start_mode_option = random.choice(START_MODE_OPTIONS)
        trial_info_dict['env_params']['start_mode'] = START_MODE_DICT[start_direction][start_mode_option]
        trial_info_dict['env_params']['location_of_turn'] = random.choice(range(1, num_turns+1))

        for j in range(num_reps_per_condition):
            with open(os.path.join(trial_dir, str(index) + '.pkl'), 'wb') as fp:
                pickle.dump(trial_info_dict, fp)
            assistance_to_pkl_index[assistance_type].append(index)
            index += 1
            print 'Trial Number ', index

    with open(os.path.join(metadata_dir, 'assistance_to_pkl_index.pkl'), 'wb') as fp:
        pickle.dump(assistance_to_pkl_index, fp)

    #create_training pickle file
    trial_info_dict = collections.OrderedDict()
    trial_info_dict['env_params'] = collections.OrderedDict()
    trial_info_dict['env_params']['r_to_g_relative_orientation'] = RGOrient.TOP_LEFT
    trial_info_dict['env_params']['num_turns'] = 1
    trial_info_dict['env_params']['start_direction'] = StartDirection.X
    trial_info_dict['env_params']['assistance_type'] = 2 #No Assistance. Just for manual practice
    trial_info_dict['env_params']['robot_position'] = ROBOT_GOAL_CONFIGURATIONS[RGOrient.TOP_LEFT]['robot']
    trial_info_dict['env_params']['robot_orientation'] = 0.0
    trial_info_dict['env_params']['goal_position'] = ROBOT_GOAL_CONFIGURATIONS[RGOrient.TOP_LEFT]['goal']
    trial_info_dict['env_params']['goal_orientation'] = PI/2

    start_mode_option = random.choice(START_MODE_OPTIONS)
    trial_info_dict['env_params']['start_mode'] = START_MODE_DICT[StartDirection.X][start_mode_option]
    trial_info_dict['env_params']['location_of_turn'] = random.choice(range(1, 2))
    with open(os.path.join(trial_dir, 'training_trial.pkl'), 'wb') as fp:
        pickle.dump(trial_info_dict, fp)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial_dir', dest='trial_dir',default=os.path.join(os.getcwd(), 'trial_dir'), help="The directory where trials will be stored are")
    parser.add_argument('--metadata_dir', dest='metadata_dir',default=os.path.join(os.getcwd(), 'metadata_dir'), help="The directory where metadata of trials will be stored")
    parser.add_argument('--num_reps_per_condition', action='store', type=int, default=3, help="number of repetetions for single combination of conditions ")
    args = parser.parse_args()
    generate_experiment_trials(args)
