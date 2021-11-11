#!/usr/bin/env python
# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

import csv
import sys
import argparse
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import bisect
import pickle
import sys
import os
import rospkg
import itertools
import collections


def read_csv_files(subject_id):

	path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'raw_data', subject_id+'_p_phm_given_phi')
	user_input_file = path + '/_joy_sip_puff.csv'
	command_prompt_file = path + '/_command_prompt.csv'
	command_prompt_df = pd.read_csv(command_prompt_file, header = 0)
	user_input_df = pd.read_csv(user_input_file, header = 0)
	return command_prompt_df, user_input_df

def ensure_ascending_time(time_stamp_array):
	for t_array in time_stamp_array:
		previous = t_array.rosbagTimestamp[0]
		for number in t_array.rosbagTimestamp:
			if number < previous:
				sys.exit('Times are not in ascending order. Fix data before proceeding')
			previous = number
    	plt.plot(range(0,len(t_array.rosbagTimestamp)), t_array.rosbagTimestamp)
    	plt.show()

def get_nearest_time_stamp(tq, time_stamp_array):
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


def get_user_response_block_indices(time_s, time_e, user_input):
	assert 'rosbagTimestamp' in user_input
	time_s_u, index_of_time_s_u = get_nearest_time_stamp(time_s, user_input.rosbagTimestamp)
	time_e_u, index_of_time_e_u = get_nearest_time_stamp(time_e, user_input.rosbagTimestamp)

	assert time_e_u > time_s_u #sanity checks
	assert index_of_time_e_u > index_of_time_s_u #sanity check

	user_response_block_indices = range(index_of_time_s_u, index_of_time_e_u) #list of indices for lookup
	return user_response_block_indices

def populate_probabilities_from_count(user_input):
	LABEL_TO_ARRAY_DICT = {'"Hard Puff"': 0, '"Hard Sip"': 1, '"Soft Puff"': 2, '"Soft Sip"':3, '"Soft-Hard Puff Deadband"': 4, '"Soft-Hard Sip Deadband"': 5} #,  '"Zero Band"': 4, }

	counts_array = np.zeros(len(LABEL_TO_ARRAY_DICT))
	length = 0

	for label in user_input:
		if label in LABEL_TO_ARRAY_DICT:
			counts_array[LABEL_TO_ARRAY_DICT[label]] += 1
			length += 1


	# for label in user_input:
	# 	if label == '"input stopped"':
	# 		pass
	# 	else:
	# 		counts_array[LABEL_TO_ARRAY_DICT[label]] += 1
	# 		length += 1

	# embed(banner1='check counts')
	if length != 0:
		norm_counts_array = counts_array/length
	else:
		norm_counts_array = counts_array = np.zeros(len(LABEL_TO_ARRAY_DICT))
	# norm_counts_array = counts_array/length

	return norm_counts_array

def _combine_probabilities(command_to_array_dict, combination_pairs):
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
# def _combine_probabilities(command_probs, combination_pairs):
# 	for i in command_probs:
# 		command_probs[i][0] = command_probs[i][0]+command_probs[i][5]
# 		command_probs[i][1] = command_probs[i][1]+command_probs[i][6]
# 		my_list = list(command_probs[i])
# 		my_list.pop(6)
# 		my_list.pop(5)
# 		command_probs[i] = np.array(my_list)
# 	return command_probs

def _init_p_um_given_ui(commands, keys, subject_id):

	p_um = collections.OrderedDict()
	for i in commands:
		p_um[i] = collections.OrderedDict()
		for index, name in enumerate(keys):
			p_um[i][name] = commands[i][index]
	# embed(banner1="u_m")
	personalized_distributions_dir = os.path.join(rospkg.RosPack().get_path('inference_engine'), 'personalized_distributions')
	print(p_um)
	pickle.dump(p_um, open(os.path.join(personalized_distributions_dir, subject_id +'_p_phm_given_phi.pkl'), "wb"))


def build_probabilities(command_prompt, user_input, subject_id):

	# h_p, h_s, s_p, s_s, s-h_p, s-h_s
	u_hp = np.zeros(6) #TODO avoid hard coding the dimensionality of these array
	u_hs = np.zeros(6)
	u_sp = np.zeros(6)
	u_ss = np.zeros(6)

	u_hp_profile = []
	u_hs_profile = []
	u_sp_profile = []
	u_ss_profile = []

	COMMAND_TO_ARRAY_DICT = collections.OrderedDict()
	COMMAND_TO_ARRAY_DICT = {'Hard Puff': u_hp, 'Hard Sip': u_hs, 'Soft Puff': u_sp, 'Soft Sip': u_ss}
	# COMMAND_TO_PROFILE_ARRAY_DICT = {'Hard Puff': u_hp_profile, 'Hard Sip': u_hs_profile, 'Soft Puff': u_sp_profile, 'Soft Sip': u_ss_profile}
	NUM_TIMES_COMMAND_PROMPT_SHOWN = {'Hard Puff': 0, 'Hard Sip': 0, 'Soft Puff': 0, 'Soft Sip': 0}
	for i in range(0,len(command_prompt)-1,2):
		key =  command_prompt.at[i, 'command'].replace('"','') #get the actual text shown on the screen.
		comm_start_t =  command_prompt.at[i, 'rosbagTimestamp'] #timestamp at which the text was shown on the screen
		comm_end_t =  command_prompt.at[i+1, 'rosbagTimestamp'] #timestamp at which the text disappeared from the screen

		user_response_block_indices = get_user_response_block_indices(comm_start_t, comm_end_t, user_input)

		user_response_header = user_input['frame_id'][user_response_block_indices]
		# embed(banner1='check indices')
		prob_count = populate_probabilities_from_count(user_response_header)

		if np.sum(prob_count == np.zeros((6,))) != 6:
			COMMAND_TO_ARRAY_DICT[key] += prob_count
			NUM_TIMES_COMMAND_PROMPT_SHOWN[key] = NUM_TIMES_COMMAND_PROMPT_SHOWN[key] + 1

		# a = user_input['axes'][user_response_block_indices].str.replace(r"\[", "")
		# a = a.str.replace(r"\]", "")
		# a = [float(i) for i in a]
		# COMMAND_TO_PROFILE_ARRAY_DICT[key].append(a)

	# embed(banner1='check after loop')

	for k, v in COMMAND_TO_ARRAY_DICT.items():
		v = v/NUM_TIMES_COMMAND_PROMPT_SHOWN[k]
		COMMAND_TO_ARRAY_DICT[k] = v
		COMMAND_TO_ARRAY_DICT[k] = v/np.sum(v) #sometimes normalization is not perfect, so do it again.



	# embed(banner1='check dict')

	# keys = ['Hard Puff', 'Hard Sip', 'Soft Puff', 'Soft Sip', 'Zero Band', 'Soft-Hard Puff Deadband', 'Soft-Hard Sip Deadband']
	# keys = ['Hard Puff', 'Hard Sip', 'Soft Puff', 'Soft Sip', 'Zero Band']
	keys = ['Hard Puff', 'Hard Sip', 'Soft Puff', 'Soft Sip']
	combination_pairs = [[0, 4], [1, 5]] #TODO replace it with dictionary mapping command strings to indices
	COMMAND_TO_ARRAY_DICT = _combine_probabilities(COMMAND_TO_ARRAY_DICT, combination_pairs)
	noise = 0.01
	# print(COMMAND_TO_ARRAY_DICT)
	for k, v in COMMAND_TO_ARRAY_DICT.items():
		v = v + noise*np.ones((4,))
		v = v/np.sum(v)
		COMMAND_TO_ARRAY_DICT[k] = v

	# embed(banner1='check dict')

	_init_p_um_given_ui(COMMAND_TO_ARRAY_DICT, keys, subject_id)

	# for i in range(len(COMMAND_TO_PROFILE_ARRAY_DICT.keys())):
	# 	plot_response_curves(COMMAND_TO_PROFILE_ARRAY_DICT[i], i)
	# plot_response_curves(u_hp_profile, 'Hard Puff')
	# plot_response_curves(u_hs_profile, 'Hard Sip')
	# plot_response_curves(u_sp_profile, 'Soft Puff')
	# plot_response_curves(u_ss_profile, 'Soft Sip')

	return u_hp, u_hs, u_sp, u_ss


def plot_response_curves(user_input, title):
	plt.xlabel("X-axis")
	plt.ylabel("user input")
	plt.title(title)
	plt.ylim(bottom=-1, top=1)
	i = 1
	for y in user_input:
		plt.plot(range(len(y)),y,label = 'id %s'%i)
		i += 1
	plt.legend()
	plt.show()
	# plt.show(block=False)
	# plt.pause(0.0001)

# def plot_timestamps():

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-id', help='subject id', type=str)
	args = parser.parse_args()
	subject_id = args.id

	topics = read_csv_files(subject_id)
	build_probabilities(topics[0], topics[1], subject_id)
