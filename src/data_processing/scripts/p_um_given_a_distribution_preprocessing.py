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
sys.path.append(os.path.join(rospkg.RosPack().get_path('simulators'), 'scripts'))
from adaptive_assistance_sim_utils import TRUE_ACTION_TO_COMMAND

class DataParser(object):
	def __init__(self, file_dir):
		super(DataParser, self).__init__()

		# results_files = os.listdir(file_dir)
		full_file_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'raw_data', file_dir)
		results_files = os.listdir(full_file_dir)
		action_prompt_file = os.path.join(full_file_dir, '_action_prompt.csv')
		user_response_file = os.path.join(full_file_dir, '_joy_sip_puff.csv')
		trial_marker_file = os.path.join(full_file_dir, '_trial_marker.csv')

		self.action_prompt_df = self.read_csv_files(action_prompt_file)
		self.user_response_df = self.read_csv_files(user_response_file)
		self.trial_marker_df = self.read_csv_files(trial_marker_file)

	def read_csv_files(self, file_path):

		df = pd.read_csv(file_path, header = 0)
		return df

class MeasuredCommandGivenActionAnalysis(object):
	def __init__(self, args):

		self.id = args.id
		self.file_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'raw_data', self.id+'_p_um_given_a')

		self.data = DataParser(self.file_dir)

	def get_nearest_time_stamp(self, tq, time_stamp_array):
		'''
		assumes that time_stamp_array is a sorted (ascending) list.
		returns: (time stamp closest to tq, index of the time stamp closest to tq in time_stamp_array)
		'''
		idx = bisect.bisect(time_stamp_array, tq) #idx is the index in the time_stamp_array whose value is immediately greater than tq
		if idx == 0: #if requested timestamp is smaller than the first element of the list then return the first element of the list and the index is 0
			return time_stamp_array[idx], idx
		elif idx == len(time_stamp_array): #if tq is greater than the biggest timestamp in the list, then retirn the last timestamp and the index of len-1 or idx-1
			return time_stamp_array[idx-1], idx-1
		else:
			prev_t = time_stamp_array[idx - 1]
			next_t = time_stamp_array[idx]
			if next_t - tq <= tq - prev_t: #if tq is closer to the next value
				return time_stamp_array[idx], idx
			else: #tq is closer to the previous value
				return time_stamp_array[idx-1], idx-1


	def get_user_response_block_indices(self, time_s, time_e, user_input):

		assert 'rosbagTimestamp' in user_input
		time_s_u, index_of_time_s_u = self.get_nearest_time_stamp(time_s, user_input.rosbagTimestamp)
		time_e_u, index_of_time_e_u = self.get_nearest_time_stamp(time_e, user_input.rosbagTimestamp)

		assert time_e_u >= time_s_u #sanity checks
		assert index_of_time_e_u >= index_of_time_s_u #sanity check

		user_response_block_indices = range(index_of_time_s_u, index_of_time_e_u) #list of indices for lookup
		return user_response_block_indices


	def populate_probabilities_from_count(self, user_input):
		LABEL_TO_ARRAY_DICT = {'"Hard Puff"': 0, '"Hard Sip"': 1, '"Soft Puff"': 2, '"Soft Sip"':3, '"Soft-Hard Puff Deadband"': 4, '"Soft-Hard Sip Deadband"': 5} #,  '"Zero Band"': 4, }

		counts_array = np.zeros(len(LABEL_TO_ARRAY_DICT))
		length = 0

		for label in user_input:
			if label in LABEL_TO_ARRAY_DICT:
				counts_array[LABEL_TO_ARRAY_DICT[label]] += 1
				length += 1

		if length != 0:
			norm_counts_array = counts_array/length
		else:
			norm_counts_array = counts_array = np.zeros(len(LABEL_TO_ARRAY_DICT))

		return norm_counts_array

	def _combine_probabilities(self, command_to_array_dict, combination_pairs):
		for key in command_to_array_dict:
			for cp in combination_pairs:
				command_to_array_dict[key][cp[0]] = command_to_array_dict[key][cp[0]] + command_to_array_dict[key][cp[1]]
			temp_list_for_popping_combined_index = list(command_to_array_dict[key])
			indices_to_be_popped = [cp[1] for cp in combination_pairs]
			indices_to_be_popped = sorted(indices_to_be_popped, reverse=True)
			for ind in indices_to_be_popped:
				temp_list_for_popping_combined_index.pop(ind)
			command_to_array_dict[key] = np.array(temp_list_for_popping_combined_index)

		return command_to_array_dict
#
	def build_distributions(self):

		# create dictionary to map user input to the low level commands for the final p_ui_given_a dict
		USER_RESPONSE_DICT = {'1': 'Hard Puff', '2': 'Hard Sip', '3': 'Soft Puff', '4': 'Soft Sip'}

		# hard puff, hard sip, soft puff, soft sip
		up = np.zeros(6)
		down = np.zeros(6)
		right = np.zeros(6)
		left = np.zeros(6)
		ccw = np.zeros(6)
		cw = np.zeros(6)
		mode_r_x = np.zeros(6)
		mode_r_y = np.zeros(6)
		mode_r_t = np.zeros(6)
		mode_l_x = np.zeros(6)
		mode_l_y = np.zeros(6)
		mode_l_t = np.zeros(6)

		# because there are x2 as many prmpts (the actual command and the empty command after)
		# and 12 total actions, so the remaining is the number of times each action was shown since all shown equally
		iters_per_action = len(self.data.action_prompt_df)/24

		# dictionary for mapping action prompts to the arrays we want to fill
		ACTION_TO_ARRAY_DICT = {'UP': up, 'DOWN': down, 'LEFT': left, 'RIGHT': right, 'CLOCKWISE': cw, 'COUNTERCLOCKWISE': ccw,
								'xy': mode_r_x, 'yt': mode_r_y, 'tx': mode_r_t,
								'xt': mode_l_x, 'yx': mode_l_y, 'ty': mode_l_t}

		# keep dict of lengths, to reduce normalizer if person missed input
		ACTION_TO_ARRAY_NORMALIZER_DICT = {'UP': iters_per_action, 'DOWN': iters_per_action, 'LEFT': iters_per_action, 'RIGHT': iters_per_action, 'CLOCKWISE': iters_per_action, 'COUNTERCLOCKWISE': iters_per_action,
								'xy': iters_per_action, 'yt': iters_per_action, 'tx': iters_per_action,
								'xt': iters_per_action, 'yx': iters_per_action, 'ty': iters_per_action}

		NUM_TIMES_ACTION_PROMPT_SHOWN = {'UP': 0, 'DOWN': 0, 'LEFT': 0, 'RIGHT': 0, 'CLOCKWISE': 0, 'COUNTERCLOCKWISE': 0,
								'xy': 0, 'yt': 0, 'tx': 0,
								'xt': 0, 'yx': 0, 'ty': 0}

		# for i in range(start_ind, len(self.data.trial_marker_df)-1, 2):

		if self.data.trial_marker_df.at[0, 'data'] == "end":
			print 'first is end:'
			ind = 1
		else:
			ind = 0


		for i in range(0, len(self.data.action_prompt_df)):

			key = self.data.action_prompt_df.at[i, 'command'].replace('"', '')

			prompt_t_s = self.data.trial_marker_df.at[ind, 'rosbagTimestamp']
			prompt_t_e = self.data.trial_marker_df.at[ind+1, 'rosbagTimestamp']

			ind += 2

			user_response_block_indices = self.get_user_response_block_indices(prompt_t_s, prompt_t_e, self.data.user_response_df)

			user_response_header = self.data.user_response_df['frame_id'][user_response_block_indices]
			prob_count = self.populate_probabilities_from_count(user_response_header)
			if np.sum(prob_count == np.zeros((6,))) != 6:
				ACTION_TO_ARRAY_DICT[key] += prob_count
				NUM_TIMES_ACTION_PROMPT_SHOWN[key] = NUM_TIMES_ACTION_PROMPT_SHOWN[key] + 1


		for k, v in ACTION_TO_ARRAY_DICT.items():
			v = v/NUM_TIMES_ACTION_PROMPT_SHOWN[k]
			ACTION_TO_ARRAY_DICT[k] = v
			ACTION_TO_ARRAY_DICT[k] = v/np.sum(v) #sometimes normalization is not perfect, so do it again.


		keys = ['Hard Puff', 'Hard Sip', 'Soft Puff', 'Soft Sip']
		combination_pairs = [[0, 4], [1, 5]] #TODO replace it with dictionary mapping command strings to indices
		# embed(banner1='check pum_given_a')
		ACTION_TO_ARRAY_DICT = self._combine_probabilities(ACTION_TO_ARRAY_DICT, combination_pairs)

		self.create_p_um_given_a(ACTION_TO_ARRAY_DICT)


		# 	if user_response_block_indices != []: # if they gave a response
		# 		user_response = int(self.data.user_response_df['command'][user_response_block_indices].replace('"', ''))
		# 		ACTION_TO_ARRAY_DICT[key][user_response-1] += 1

		# 	else:
		# 		ACTION_TO_ARRAY_NORMALIZER_DICT[key] -= 1


		# # normalize
		# for k, v in ACTION_TO_ARRAY_DICT.items():
		# 	v = v/ACTION_TO_ARRAY_NORMALIZER_DICT[k]
		# 	ACTION_TO_ARRAY_DICT[k] = v

		# self.create_p_ui_given_a(ACTION_TO_ARRAY_DICT)

	def create_p_um_given_a(self, probabilities):
		keys = ['Hard Puff', 'Hard Sip', 'Soft Puff', 'Soft Sip']
		p_um = collections.OrderedDict()
		for mode in TRUE_ACTION_TO_COMMAND.keys():
			p_um[mode] = collections.OrderedDict()
			for action in TRUE_ACTION_TO_COMMAND[mode].keys():
				p_um[mode][action] = collections.OrderedDict()
				if mode == 'x':
					if action == 'move_p':
						prob = probabilities['RIGHT']
					if action == 'move_n':
						prob = probabilities['LEFT']
					if action == 'mode_r':
						prob = probabilities['xy']
					if action == 'mode_l':
						prob = probabilities['xt']
				if mode == 'y':
					if action == 'move_p':
						prob = probabilities['UP']
					if action == 'move_n':
						prob = probabilities['DOWN']
					if action == 'mode_r':
						prob = probabilities['yt']
					if action == 'mode_l':
						prob = probabilities['yx']
				if mode == 't':
					if action == 'move_p':
						prob = probabilities['COUNTERCLOCKWISE']
					if action == 'move_n':
						prob = probabilities['CLOCKWISE']
					if action == 'mode_r':
						prob = probabilities['tx']
					if action == 'mode_l':
						prob = probabilities['ty']

				# noise_level = 0.01
				# prob = prob + noise_level*np.ones((4,))
				# prob = prob/np.sum(prob)

				for ind, key in enumerate(keys):
					p_um[mode][action][key] = prob[ind]

		personalized_distributions_dir = os.path.join(rospkg.RosPack().get_path('inference_and_correction'), 'personalized_distributions')
		pickle.dump(p_um, open(os.path.join(personalized_distributions_dir,self.id+'_p_um_given_a.pkl'), "wb"))


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-id', help='subject id', type=str)
	args = parser.parse_args()
	# embed(banner1="before initialization")
	puma = MeasuredCommandGivenActionAnalysis(args)
	puma.build_distributions()

	# python p_um_given_a_distribution_preprocessing.py -path deepak_p_um_given_a -id deepak
