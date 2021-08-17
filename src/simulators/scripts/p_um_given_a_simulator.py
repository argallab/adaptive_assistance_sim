#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

import rospy
import time
from sensor_msgs.msg import Joy
from simulators.msg import Command
from envs.p_um_given_a_env import PUmGivenAEnv
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension, String, Int8
from teleop_nodes.msg import CartVelCmd
from teleop_nodes.msg import ModeSwitch
from simulators.msg import State
from teleop_nodes.srv import SetMode, SetModeRequest, SetModeResponse
from pyglet.window import key
import numpy as np
import pickle
from pyglet.window import key
import collections
import random
import sys
import os
from corrective_mode_switch_utils import Robot2D, FPS, VELOCITY_ITERATIONS, POSITION_ITERATIONS, SCALE, VIEWPORT_W, VIEWPORT_H, DIM_TO_MODE_INDEX, VIEWPORT_WS, VIEWPORT_HS
from corrective_mode_switch_utils import ROBOT_RADIUS, GOAL_RADIUS, GOAL_SHAPES, GOAL_COLORS, PI, HUMAN_ROBOT_COLOR, AUTONOMY_ROBOT_COLOR, TRIANGLE_L
from corrective_mode_switch_utils import RGOrient, StartDirection, AssistanceType


class Simulator(object):
    def __init__(self, subject_id, training):
        super(Simulator, self).__init__()
        rospy.init_node("Simulator")
        rospy.on_shutdown(self.shutdown_hook)
        self.called_shutdown = False

        self.dim = 3
        user_vel = CartVelCmd()
        _dim = [MultiArrayDimension()]
        _dim[0].label = 'cartesian_velocity'
        _dim[0].size = self.dim
        _dim[0].stride = self.dim
        user_vel.velocity.layout.dim = _dim
        user_vel.velocity.data = np.zeros(self.dim)
        user_vel.header.stamp = rospy.Time.now()
        user_vel.header.frame_id = 'human_control'

        self.action_msg = Command()

        self.trial_info_dir_path = os.path.join(os.path.dirname(__file__), 'p_um_given_a_trial_dir') #this directory will be populated by generate_p_um_given_a_trials.py
        self.metadata_dir = os.path.join(os.path.dirname(__file__), 'metadata_dir')
        self.trial_list = None
        self.input_action = {}
        self.input_action['human'] = user_vel

        self.trial_index = 0
        self.env_params = None
        self.subject_id = subject_id
        self.training = training
        self.terminate = False
        self.restart = False

        self.allowed_mode_index_dict = {StartDirection.X: 'x', StartDirection.Y: 'y'}
        self.goal_location_relative_to_robot_location = {StartDirection.X: {'positive':[VIEWPORT_WS/4, 0], 'negative':[-VIEWPORT_WS/4, 0]},
                                                         StartDirection.Y: {'positive':[0, VIEWPORT_HS/4], 'negative':[0, -VIEWPORT_HS/4]}}

        self.initialize_publishers()
        self.initialize_subscribers()

        if self.trial_info_dir_path is not None and os.path.exists(self.trial_info_dir_path):
            self.trial_list = os.listdir(self.trial_info_dir_path)
            random.shuffle(self.trial_list) #shuffle the trial list for each subject
            trial_info_filename = self.trial_list[self.trial_index] #already as the .pkl extensions since the list was created from os.listdir
            #load up trial info
            trial_info_filepath = os.path.join(self.trial_info_dir_path, trial_info_filename)
            assert os.path.exists(trial_info_filepath) is not None
            with open(trial_info_filepath, 'rb') as fp:
                trial_info_dict = pickle.load(fp)

            assert 'env_params' in trial_info_dict
            self.env_params = trial_info_dict['env_params']

        else:
            #if there are no trials in trials folder, use this trial info.
            trial_info_filename = 'default.pkl' #dummy filename for publishing
            self.env_params = dict()
            self.env_params['robot_position'] = (VIEWPORT_WS/2, VIEWPORT_HS/2) #robot position will be in the center
            self.env_params['robot_orientation'] = 0.0 #robot orientation will always be 0.0
            self.env_params['start_direction'] = StartDirection.X #which direction is the path pointing.
            self.env_params['is_rotation'] = False #whether this trial has rotation as opposed to linear motion
            self.env_params['is_mode_switch'] = False #whether this trial is testing hard puffs or sips.
            start_direction = self.env_params['start_direction']

            self.env_params['goal_position'] = list(np.array(self.env_params['robot_position']) + np.array(self.goal_location_relative_to_robot_location[start_direction]['positive']))
            if not self.env_params['is_rotation']: #conditioned on whether it is rotation or not specify which mode is allowed mode for movement and also the goal orientation
                self.env_params['allowed_mode_index'] = self.allowed_mode_index_dict[self.env_params['start_direction']]
                self.env_params['goal_orientation'] = self.env_params['robot_orientation']
            else:
                self.env_params['allowed_mode_index'] = 't'
                self.env_params['goal_orientation'] = PI/2

            if self.env_params['is_mode_switch']: #if this trial is about mode switching, the environment will not display any robot or goal, but instead will just display the mode circle.
                self.mode_config = collections.OrderedDict()
                self.mode_config['start_mode'] = 'x' #specify the starting mode
                self.mode_config['target_mode'] = 't' #specify the target mode
                self.env_params['mode_config'] = self.mode_config
            else:
                self.env_params['mode_config'] = None

        #Set the proper starting mode in the teleop node using the set mode service.
        rospy.loginfo("Waiting for teleop_node ")
        rospy.wait_for_service("/teleop_node/set_mode")
        rospy.loginfo("teleop_node node service found! ")

        self.set_mode_srv = rospy.ServiceProxy('/teleop_node/set_mode', SetMode)
        self.set_mode_request = SetModeRequest()
        self.set_mode_request.mode_index = DIM_TO_MODE_INDEX[self.env_params['allowed_mode_index']]
        status = self.set_mode_srv(self.set_mode_request)

        #instantiate the environment
        self.env_params['start'] = False
        self.env = PUmGivenAEnv(self.env_params)
        self.env.initialize()
        self.env.initialize_viewer()
        self.env.viewer.window.on_key_press = self.key_press

        self.start = False
        first_trial = True
        is_done = False

        r = rospy.Rate(100)
        if self.training:
            self.max_time = 1000
        else:
            self.max_time = 5

        self.max_time = 5 # TODO change this for controlling max time for showing each prompt
        if self.training:
            time_check = self.max_time - 1

        while not rospy.is_shutdown():
            if not self.start:
                self.start = self.env.start_countdown()
            else:
                if first_trial:
                    time.sleep(2)
                    self.trial_marker_pub.publish('start')
                    self.publish_action(self.env_params['action'])
                    self.trial_filename_pub.publish(trial_info_filename)
                    self.env.reset()
                    self.env.render()
                    self.trial_start_time = time.time()
                    first_trial = False
                else:
                    if not self.training:
                        time_check = (time.time() - self.trial_start_time)
                    if time_check > self.max_time or is_done:
                        print("Move to NEXT TRIAL")
                        self.trial_marker_pub.publish('end')
                        #clear screen
                        if self.env_params['is_mode_switch']:
                            time.sleep(0.5)
                        self.env.render_clear('Loading next trial ...')
                        time.sleep(1.0)
                        self.trial_index += 1
                        if self.trial_index == len(self.trial_list):
                            self.shutdown_hook('Reached end of trial list. End of Session')
                            break
                        trial_info_filename = self.trial_list[self.trial_index]
                        trial_info_filepath = os.path.join(self.trial_info_dir_path, trial_info_filename)
                        assert os.path.exists(trial_info_filepath) is not None
                        with open(trial_info_filepath, 'rb') as fp:
                            trial_info_dict = pickle.load(fp)
                        assert 'env_params' in trial_info_dict
                        self.env_params = trial_info_dict['env_params']
                        self.set_mode_request = SetModeRequest()
                        self.set_mode_request.mode_index = DIM_TO_MODE_INDEX[self.env_params['allowed_mode_index']]
                        status = self.set_mode_srv(self.set_mode_request)

                        self.trial_marker_pub.publish('start')
                        self.publish_action(self.env_params['action'])
                        self.trial_filename_pub.publish(trial_info_filename)

                        self.env.update_params(self.env_params)
                        self.env.reset()
                        self.env.render()

                        self.trial_start_time = time.time()
                        self.restart = False

                if self.restart:
                    self.trial_marker_pub.publish('restart')
                    self.restart = False
                    time.sleep(2.0)
                    self.trial_index += 1
                    if self.trial_index == len(self.trial_list):
                        self.shutdown_hook('Reached end of trial list. End of Session')
                        break
                    trial_info_filename = self.trial_list[self.trial_index]
                    trial_info_filepath = os.path.join(self.trial_info_dir_path, trial_info_filename)
                    assert os.path.exists(trial_info_filepath) is not None
                    with open(trial_info_filepath, 'rb') as fp:
                        trial_info_dict = pickle.load(fp)

                    assert 'env_params' in trial_info_dict
                    self.env_params = trial_info_dict['env_params']
                    self.set_mode_request = SetModeRequest()
                    self.set_mode_request.mode_index = DIM_TO_MODE_INDEX[self.env_params['allowed_mode_index']]
                    status = self.set_mode_srv(self.set_mode_request)

                    self.trial_marker_pub.publish('start')
                    self.publish_action(self.env_params['action'])
                    self.trial_filename_pub.publish(trial_info_filename)

                    self.env.update_params(self.env_params)
                    self.env.reset()
                    self.env.render()

                    self.trial_start_time = time.time()
                    self.restart = False

                if self.terminate:
                    self.shutdown_hook('Session terminated')
                    break

                is_done = self.env.step(self.input_action)
                self.env.render()

            r.sleep()

    def joy_callback(self, msg):
        self.input_action['human'] = msg

    def mode_switch_callback(self, msg):
        if not self.env_params['is_mode_switch']:
            if msg.mode != DIM_TO_MODE_INDEX[self.env_params['allowed_mode_index']]:
                status = self.set_mode_srv(self.set_mode_request)

    def shutdown_hook(self, msg_string='DONE'):
        if not self.called_shutdown:
            self.called_shutdown = True
            self.shutdown_pub.publish("shutdown")
            #clear screen
            self.env.render_clear('End of trial...')
            self.env.close_window()
            print('Shutting down')

    def key_press(self, k, mode):
        if k == key.SPACE:
            self.terminate = True
        if k == key.R:
            self.restart = True

    def initialize_publishers(self):
        self.shutdown_pub = rospy.Publisher('/shutdown', String, queue_size=1)
        self.trial_marker_pub = rospy.Publisher('/trial_marker', String, queue_size=1)
        self.trial_filename_pub = rospy.Publisher('/trial_index', String, queue_size=1)
        self.action_pub = rospy.Publisher('/action_prompt', Command, queue_size=1)

    def initialize_subscribers(self):
        rospy.Subscriber('/user_vel', CartVelCmd, self.joy_callback)
        rospy.Subscriber('/mode_switches', ModeSwitch, self.mode_switch_callback)

    def publish_action(self, msg):
        self.action_msg.header.stamp = rospy.Time.now()
        self.action_msg.command = msg
        self.action_pub.publish(self.action_msg)

if __name__ == '__main__':
    subject_id = sys.argv[1]
    training = int(sys.argv[2])
    Simulator(subject_id, training)
    rospy.spin()
