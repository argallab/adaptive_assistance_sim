#!/usr/bin/env python

# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

# This is a python script for extracting topics from a rosbag.
# It coverts each topic from the bag into a pandas dataframe
# requirements:  pip install rosbag_pandas

import rosbag
import rosbag_pandas

from rospy.numpy_msg import numpy_msg


# input to this script is the path and file name, topics to extract

bag = rosbag.Bag(bag_name)


msgs = bag.read_messages(topics=topic_list[])

class ROSBag(object):

	def __init__(self, bag_names, topics=None, path, save_prefix=None):

		# initialize
		self.bags = self.get_bags(bag_names)
		self.topics = topics
		self.save_prefix = save_prefix

		#

	# close bags after destruction
	def __del__(self):
		self.close_bags()

	def close_bag(self):
		for bag in self.bags:
			bag.close()



# bag = rosbag.Bag('test.bag')
# for topic, msg, t in bag.read_messages(topics=['chatter', 'numbers']):
#     print(msg)
# bag.close()



# # Get lists of topics and types from bag file
# import rosbag
# bag = rosbag.Bag('input.bag')
# topics = bag.get_type_and_topic_info()[1].keys()
# types = []
# for i in range(0,len(bag.get_type_and_topic_info()[1].values())):
#     types.append(bag.get_type_and_topic_info()[1].values()[i][0])


# # get summary information about a bag:
# import yaml
# from rosbag.bag import Bag

# info_dict = yaml.load(Bag('input.bag', 'r')._get_yaml_info())
