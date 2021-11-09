#!/usr/bin/env python
# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

import os
import argparse
import pandas as pd
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

		# get path to block of interest
		self.block = args.block
		self.block_name_info = self.block.partition('_')
		self.block_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'raw_data', self.block)


		assert os.path.exists(self.block_dir)

		# To Do: make these inputs to function
		self.topics = {	'infer_correct_info': '_infer_correct_info.csv',
						'joy_sip_puff': '_joy_sip_puff.csv',
						'joy_sip_puff_before': '_joy_sip_puff_before.csv',
						'mode_switches': '_mode_switches.csv',
						'robot_state': '_robot_state.csv',
						'trial_index': '_trial_index.csv',
						'trial_marker': '_trial_marker.csv',
						'user_vel': '_user_vel.csv' }

		self.sub_topics = {'infer_correct_info':['rosbagTimestamp', 'optimal_a', 'u_intended', 'normalized_h', 'u_corrected', 'is_corrected_or_filtered', 'is_u_intended_equals_um'],
						  'joy_sip_puff': ['rosbagTimestamp', 'frame_id', 'axes', 'buttons'],
						  'joy_sip_puff_before': ['rosbagTimestamp', 'frame_id'],
						  'mode_switches':['rosbagTimestamp', 'mode', 'frame_id'],
						  'robot_state': ['rosbagTimestamp', 'robot_continuous_position', 'robot_continuous_orientation', 'robot_linear_velocity', 'robot_angular_velocity', 'robot_discrete_state', 'discrete_location', 'discrete_orientation', 'mode'],
						  'trial_index': ['rosbagTimestamp', 'data'],
						  'trial_marker': ['rosbagTimestamp', 'data'],
						  'user_vel': ['rosbagTimestamp', 'data']}



	def load_topic_csvs(self):
		# read in topic csvs of interest with name and index

		# create dict to store topics as variables in loop
		self.df_dict = {}

		for t in self.topics.keys():

			folder = os.path.join(self.block_dir, self.topics[t])

			if os.path.exists(folder):
				print('Loading '+t)

				df = pd.read_csv(folder, header = 0, usecols=self.sub_topics[t])

				keys = [key for key in dict(df.dtypes) if dict(df.dtypes)[key] in ['object']]
				for key in keys:
					df[key].apply(lambda s:s.replace('"', "")) # remove double quotation
					df[key] = df[key].apply(literal_eval) #convert data types in [] where some rows are zero (considers mixed so dtype is object instead of int or float array)

				assert self.ensure_ascending(df.rosbagTimestamp) # sanity check, ensure rosbag times in ascending order
				self.df_dict[t] = df
				# print self.df_dict[t].dtypes

			else:
				print ('[warning] '+t+' does not exist!!!') # for example, no_assistance wouldn't ahve infer_correct_infor. that is fine

		assert self.ensure_all_trials_marked()	# sanity check


	def ensure_ascending(self, data):

		flag = 0
		if (all(data[i] <= data[i+1] for i in range(len(data)-1))):
			flag = 1
		return flag


	def ensure_all_trials_marked(self):

		flag = 1
		# make sure have the right number of start and end markers
		self.num_trials = len(self.df_dict['trial_index'])
		start_markers = self.df_dict['trial_marker'].loc[self.df_dict['trial_marker']['data']=='start']
		end_markers = self.df_dict['trial_marker'].loc[self.df_dict['trial_marker']['data']=='end']
		restart_markers = self.df_dict['trial_marker'].loc[self.df_dict['trial_marker']['data']=='restart']

		if len(start_markers) != self.num_trials:
			if not len(restart_markers) == (self.num_trials - len(start_markers)):
				flag = 0

		return flag

	def rename_df_column(self):
		# specific corrections:
		# robot_state/mode --> robot_state_mode
		# joy_sip_puff_before/frame_id ---> joy_sip_puff_before
		# user_vel/data --> user_vel

		# To Do: make this an input to function
		self.df_dict['joy_sip_puff'].rename(columns={'frame_id':'sip_puff_frame_id', 'axes':'sip_puff_axes',
			'buttons':'sip_puff_buttons'}, inplace=True)
		self.df_dict['joy_sip_puff_before'].rename(columns={'frame_id':'sip_puff_before_frame_id'}, inplace=True)
		self.df_dict['robot_state'].rename(columns={'mode':'robot_state_mode'}, inplace=True)
		self.df_dict['user_vel'].rename(columns={'data':'user_vel'}, inplace=True)
		self.df_dict['trial_marker'].rename(columns={'data':'trial_marker'}, inplace=True)
		self.df_dict['trial_index'].rename(columns={'data':'trial_index'}, inplace=True)
		self.df_dict['mode_switches'].rename(columns={'frame_id':'mode_frame_id'}, inplace=True)

		for df in self.df_dict.keys():
			self.df_dict[df].rename(columns={'rosbagTimestamp':'time'}, inplace=True)


	def get_trial_indices(self):
		# assumes already changed name of column
		bool_start = False
		self.start_ind = []
		self.end_ind = []
		for i in range(len(self.df_dict['trial_marker'])):
			# get start index
			if self.df_dict['trial_marker'].loc[i,'trial_marker'] == 'start' and bool_start == False:
				self.start_ind.append(self.df_dict['trial_marker'].loc[i,'time'])
				bool_start = True
			# get end index
			elif self.df_dict['trial_marker'].loc[i,'trial_marker'] == 'end' and bool_start == True:
				self.end_ind.append(self.df_dict['trial_marker'].loc[i,'time'])
				bool_start = False
			# if was reset, reset start index
			elif self.df_dict['trial_marker'].loc[i,'trial_marker'] == 'restart' and bool_start == True:
				self.start_ind.pop() # pop out last element from start, not true start
				bool_start = False
			else:
				print 'unexpected outcome'

		assert len(self.start_ind) == len(self.end_ind)


	def create_csv_per_trial(self):

		self.load_topic_csvs() # load data
		self.rename_df_column() # prevent identical column names before merging
		self.get_trial_indices() # get the time value for each start and end, will use to create trial csvs

		# get all dataframes except trial_index (doesn't need to be included, just for naming)
		keys = self.df_dict.keys()
		keys.remove('trial_index')
		frames = [self.df_dict[key] for key in keys]

		# combine all data together using the time stamp
		frankenstein_df = frames[0]
		for i in range(1,len(frames)):
			print i, frames[i].keys()
			frankenstein_df = frankenstein_df.merge(frames[i], how='outer', sort=True)

		assert self.ensure_ascending(frankenstein_df.time) # sanity check

		# path to store trial data for subject:
		trial_dir = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir)), 'data_analysis/data', self.block_name_info[0])
		try:	#else already exists
			os.makedirs(trial_dir)
		except:
			pass

		# save
		for i in range(self.num_trials):
			trial = frankenstein_df.loc[(frankenstein_df['time'] >= self.start_ind[i]) & (frankenstein_df['time'] <= self.end_ind[i])]
			# reset index for the datafram so starts from 0
			trial.reset_index(drop=True, inplace=True)
			ts = trial.loc[0, 'time']
			for j in range(len(trial)):
				trial.at[j, 'time'] = trial.loc[j, 'time'] - ts
			trial_filename = os.path.join(trial_dir, self.block+'_'+str(self.df_dict['trial_index'].loc[i, 'trial_index'])+'.csv')
			trial.to_csv(trial_filename, index=False)



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-b', '--block', help='experiment block: subject_id_type_assistance_block', type=str)
	args = parser.parse_args()
	block_to_trial = ConcatenateMainStudyTopicsPerTrial(args)
	block_to_trial.create_csv_per_trial()

	# python main_study_concatenate_topics_per_trial -block block_folder_name
