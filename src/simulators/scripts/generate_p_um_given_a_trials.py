#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

import sys
from corrective_mode_switch_utils import SCALE, VIEWPORT_W, VIEWPORT_H, ROBOT_RADIUS, GOAL_RADIUS, PI, VIEWPORT_HS, VIEWPORT_WS
from corrective_mode_switch_utils import AssistanceType, StartDirection, RGOrient
import numpy as np
import os
import pickle
import collections
import itertools
import argparse
import random

ROBOT_RADIUS_S = ROBOT_RADIUS/SCALE
GOAL_RADIUS_S = GOAL_RADIUS/SCALE

CENTER_X = VIEWPORT_WS/2
CENTER_Y = VIEWPORT_HS/2

ACTIONS = ['UP', 'DOWN', 'RIGHT', 'LEFT', 'COUNTERCLOCKWISE', 'CLOCKWISE', 'xy', 'xt', 'yx', 'yt', 'tx', 'ty']
MODE_TRANSITIONS = ['xy', 'xt', 'yx', 'yt', 'tx', 'ty']
START_DIRECTION = {'UP': [StartDirection.Y],
                   'DOWN': [StartDirection.Y],
                   'RIGHT':[StartDirection.X],
                   'LEFT': [StartDirection.X],
                   'COUNTERCLOCKWISE': [StartDirection.X, StartDirection.Y],
                   'CLOCKWISE': [StartDirection.X, StartDirection.Y]}

GOAL_CONFIGURATIONS = {'UP': [CENTER_X, CENTER_Y + CENTER_Y/2],
                   'DOWN': [CENTER_X, CENTER_Y - CENTER_Y/2],
                   'RIGHT':[CENTER_X + CENTER_X/2, CENTER_Y],
                   'LEFT': [CENTER_X - CENTER_X/2, CENTER_Y],
                   'CLOCKWISE': [CENTER_X, CENTER_Y],
                   'COUNTERCLOCKWISE': [CENTER_X, CENTER_Y]}

ALLOWED_MODE_INDEX_DICT = {StartDirection.X: 'x', StartDirection.Y: 'y'}
ANGLES = {'COUNTERCLOCKWISE': PI/2, 'CLOCKWISE': -PI/2}

def generate_p_um_given_a_trials(args):
    num_reps_per_condition = args.num_reps_per_condition
    trial_dir = args.trial_dir
    metadata_dir = args.metadata_dir
    if not os.path.exists(trial_dir):
        os.makedirs(trial_dir)

    if not os.path.exists(metadata_dir):
        os.makedirs(metadata_dir)

    index = 0
    for a in ACTIONS:#ACTIONS consist of all 12 possible actions.
        trial_info_dict = collections.OrderedDict()
        trial_info_dict['env_params'] = collections.OrderedDict()
        trial_info_dict['env_params']['robot_position'] = (VIEWPORT_WS/2, VIEWPORT_HS/2)
        trial_info_dict['env_params']['robot_orientation'] = 0.0
        trial_info_dict['env_params']['action'] = a

        if a in MODE_TRANSITIONS:  # mode switch trial
            trial_info_dict['env_params']['is_mode_switch'] = True
            trial_info_dict['env_params']['start_direction'] = random.choice(list(START_DIRECTION.values()))[0] # mode transition, so start direction doesn't matter
            mode_config = collections.OrderedDict()
            mode_config['start_mode'] = a[0] #the first mode from the string. For example 'x' in 'xy' or 't' in tx'
            mode_config['target_mode'] = a[1] #the second mode from the string. For example 'y' in 'xy' or 'x' in tx'
            trial_info_dict['env_params']['mode_config'] = mode_config
            trial_info_dict['env_params']['goal_position'] =  [CENTER_X, CENTER_Y] # doesn't matter for mode switch, has it's target mode
            trial_info_dict['env_params']['allowed_mode_index'] = a[0] # just for setting the start mode
        else: # motion trial
            trial_info_dict['env_params']['is_mode_switch'] = False
            trial_info_dict['env_params']['start_direction'] = random.choice(START_DIRECTION[a])
            trial_info_dict['env_params']['mode_config'] = None
            trial_info_dict['env_params']['goal_position'] =  GOAL_CONFIGURATIONS[a]
            trial_info_dict['env_params']['allowed_mode_index'] = ALLOWED_MODE_INDEX_DICT[trial_info_dict['env_params']['start_direction']]

        if a == 'COUNTERCLOCKWISE' or a == 'CLOCKWISE': #turning trials
            trial_info_dict['env_params']['is_rotation'] = True
            trial_info_dict['env_params']['allowed_mode_index'] = 't'
            trial_info_dict['env_params']['goal_orientation'] = ANGLES[a]

        else:
            #non turning trials
            trial_info_dict['env_params']['is_rotation'] = False
            trial_info_dict['env_params']['goal_orientation'] = 0.0

        for j in range(num_reps_per_condition):
            with open(os.path.join(trial_dir, str(index) + '.pkl'), 'wb') as fp:
                pickle.dump(trial_info_dict, fp)
            index += 1
            print 'Trial Index', index

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial_dir', dest='trial_dir',default=os.path.join(os.getcwd(), 'p_um_given_a_trial_dir'), help="The directory where trials will be stored are")
    parser.add_argument('--metadata_dir', dest='metadata_dir',default=os.path.join(os.getcwd(), 'metadata_dir'), help="The directory where metadata of trials will be stored")
    parser.add_argument('--num_reps_per_condition', action='store', type=int, default=4, help="number of repetetions for single combination of conditions ")
    args = parser.parse_args()
    generate_p_um_given_a_trials(args)
