#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

import rospy
import time
from sensor_msgs.msg import Joy
from envs.robot4d_env import Robot4DEnv
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension, String, Int8
from teleop_nodes.msg import CartVelCmd
from simulators.msg import State4D
from teleop_nodes.srv import SetMode, SetModeRequest, SetModeResponse
from pyglet.window import key
import numpy as np
import pickle
from pyglet.window import key
import random
import sys
import os
from corrective_mode_switch_utils import Robot2D, FPS, VELOCITY_ITERATIONS, POSITION_ITERATIONS, SCALE, VIEWPORT_W, VIEWPORT_H, DIM_TO_MODE_INDEX
from corrective_mode_switch_utils import ROBOT_RADIUS, GOAL_RADIUS, GOAL_SHAPES, GOAL_COLORS, PI, HUMAN_ROBOT_COLOR, AUTONOMY_ROBOT_COLOR, TRIANGLE_L
from corrective_mode_switch_utils import RGOrient, StartDirection, AssistanceType

class Simulator(object):
    def __init__(self, subject_id, assistance_block, block_id, training):
        super(Simulator, self).__init__()
        rospy.init_node("Simulator")
        rospy.on_shutdown(self.shutdown_hook)
        self.called_shutdown = False
        self.shutdown_pub = rospy.Publisher('/shutdown', String, queue_size=1)
        self.trial_marker_pub = rospy.Publisher('/trial_marker', String, queue_size=1)
        self.trial_index_pub = rospy.Publisher('/trial_index', Int8, queue_size=1)
        self.robot_state_pub = rospy.Publisher('/robot_state', State4D, queue_size=1)

        self.robot_state = State4D()
        self.dim = 4
        user_vel = CartVelCmd()
        _dim = [MultiArrayDimension()]
        _dim[0].label = 'cartesian_velocity'
        _dim[0].size = self.dim
        _dim[0].stride = self.dim
        user_vel.velocity.layout.dim = _dim
        user_vel.velocity.data = np.zeros(self.dim)
        user_vel.header.stamp = rospy.Time.now()
        user_vel.header.frame_id = 'human_control'

        stabilizing_control = CartVelCmd()
        _dim = [MultiArrayDimension()]
        _dim[0].label = 'cartesian_velocity_sc'
        _dim[0].size = self.dim
        _dim[0].stride = self.dim
        stabilizing_control.velocity.layout.dim = _dim
        stabilizing_control.velocity.data = np.zeros(self.dim)
        stabilizing_control.header.stamp = rospy.Time.now()
        stabilizing_control.header.frame_id = 'stabilizing_control'

        self.input_action = {}
        self.input_action['human'] = user_vel
        rospy.Subscriber('/user_vel', CartVelCmd, self.joy_callback)
        self.input_action['stabilizing_control'] = stabilizing_control
        rospy.Subscriber('/stabilizing_control_action', CartVelCmd, self.sc_callback)
        self.trial_index = 0

        self.env_params = None
        self.trial_info_dir_path = os.path.join(os.path.dirname(__file__), 'trial_dir_4d')
        self.metadata_dir = os.path.join(os.path.dirname(__file__), 'metadata_dir_4d')

        self.subject_id = subject_id
        self.assistance_block = assistance_block #pass these things from launch file
        self.block_id = block_id
        self.training = training
        self.total_blocks = 6

        self.testing_block_filename = self.subject_id + '_' + self.assistance_block + '_assistance_' + self.block_id + '_num_blocks_' + str(self.total_blocks) + '.pkl'
        print "TRAINING BLOCK FILENAME and IS TRAINING MODE", self.testing_block_filename, self.training

        self.terminate = False
        self.restart = False

        if self.trial_info_dir_path is not None and os.path.exists(self.trial_info_dir_path) and not self.training:
            print ('LOAD METADATA')
            self.metadata_index_path = os.path.join(self.metadata_dir, self.testing_block_filename) #metadata containing indices for the specific testing block
            assert os.path.exists(self.metadata_index_path)
            with open(self.metadata_index_path, 'rb') as fp:
                self.metadata_index = pickle.load(fp)

            trial_info_filename_index = self.metadata_index[self.trial_index]
            trial_info_filepath = os.path.join(self.trial_info_dir_path, str(trial_info_filename_index) + '.pkl')
            assert os.path.exists(trial_info_filepath) is not None
            with open(trial_info_filepath, 'rb') as fp:
                trial_info_dict = pickle.load(fp) #this dict could have other info related to autonomy params. We are only interested in the environment params for the time being

            assert 'env_params' in trial_info_dict
            trial_info_dict['env_params']['training'] = self.training
            self.env_params = trial_info_dict['env_params']
            print 'ASSISTANCE_TYPE', self.env_params['assistance_type']
        else:
            print('Create env on the fly')
            self.metadata_index = [1]
            if not self.training:
                trial_info_filename_index = 0
                self.env_params = dict()
                self.env_params['training'] = False
                self.env_params['num_turns'] = 3
                self.env_params['robot_position'] = ((VIEWPORT_W)/4/SCALE, (3*VIEWPORT_H)/4/SCALE)
                self.env_params['goal_position'] = ((3*VIEWPORT_W)/4/SCALE, (VIEWPORT_H)/4/SCALE)
                self.env_params['robot_orientation'] = 0.0
                self.env_params['goal_orientation'] = PI/2
                self.env_params['goal_gripper_angle'] = PI/8
                self.env_params['start_gripper_angle'] = PI/2
                self.env_params['r_to_g_relative_orientation'] = RGOrient.BOTTOM_RIGHT
                self.env_params['start_direction'] = StartDirection.Y
                self.env_params['start_mode'] = 't'
                self.env_params['location_of_turn'] = 2
                self.env_params['location_of_gripper_action'] = 3
                self.env_params['assistance_type'] = 1
                assert self.env_params['location_of_turn'] > 0 and self.env_params['location_of_turn'] <= self.env_params['num_turns'] #can't be the first or last location
                assert self.env_params['location_of_gripper_action'] > 0 and self.env_params['location_of_gripper_action'] <= self.env_params['num_turns']
                assert self.env_params['location_of_gripper_action'] != self.env_params['location_of_turn'] #turning and gripping locatins don't overlap
            else:
                print ('LOADING TRAINING TRIAL')
                trial_info_filename_index = 100
                assert os.path.exists(self.trial_info_dir_path)
                trial_info_filepath = os.path.join(self.trial_info_dir_path, 'training_trial.pkl')
                assert os.path.exists(trial_info_filepath) is not None
                with open(trial_info_filepath, 'rb') as fp:
                    trial_info_dict = pickle.load(fp) #this dict could have other info related to autonomy params. We are only interested in the environment params for the time being

                assert 'env_params' in trial_info_dict
                trial_info_dict['env_params']['training'] = self.training
                self.env_params = trial_info_dict['env_params']
                print  'ASSISTANCE_TYPE', self.env_params['assistance_type']

        rospy.set_param('assistance_type', self.env_params['assistance_type'])
        rospy.loginfo("Waiting for teleop_node ")
        rospy.wait_for_service("/teleop_node/set_mode")
        rospy.loginfo("teleop_node node service found! ")

        # #set starting mode for the trial
        self.set_mode_srv = rospy.ServiceProxy('/teleop_node/set_mode', SetMode)
        self.set_mode_request = SetModeRequest()
        self.set_mode_request.mode_index = DIM_TO_MODE_INDEX[self.env_params['start_mode']]
        status = self.set_mode_srv(self.set_mode_request)

        # instantiate the environement
        self.env_params['start'] = False
        self.env = Robot4DEnv(self.env_params)
        self.env.initialize()
        self.env.initialize_viewer()
        self.env.viewer.window.on_key_press = self.key_press

        r = rospy.Rate(100)
        self.trial_start_time = time.time()
        if not self.training:
            self.max_time = 200
        else:
            self.max_time = 1000
        is_done = False
        first_trial = True
        self.start = False

        while not rospy.is_shutdown():
            if not self.start:
                self.start = self.env.start_countdown()
            else:
                if first_trial: #if first trial in a block add an extra 2 seconds so that the subject is not surprised
                    time.sleep(2)
                    self.trial_marker_pub.publish('start')
                    self.trial_index_pub.publish(trial_info_filename_index)
                    self.env.reset()
                    self.env.render()
                    self.trial_start_time = time.time()
                    first_trial = False
                else:
                    if (time.time() - self.trial_start_time) > self.max_time or is_done:
                        if not self.training:
                            print("Move to NEXT TRIAL")
                            self.trial_marker_pub.publish('end')
                            self.env.render_clear('Loading next trial ...')
                            time.sleep(5.0) #sleep before the next trial happens
                            self.trial_index += 1
                            if self.trial_index == len(self.metadata_index):
                                self.shutdown_hook('Reached end of trial list. End of session')
                                break #experiment is done
                            trial_info_filename_index = self.metadata_index[self.trial_index]
                            trial_info_filepath = os.path.join(self.trial_info_dir_path, str(trial_info_filename_index) +'.pkl')
                            assert os.path.exists(trial_info_filepath) is not None
                            with open(trial_info_filepath, 'rb') as fp:
                                trial_info_dict = pickle.load(fp)

                            assert 'env_params' in trial_info_dict
                            trial_info_dict['env_params']['training'] = self.training
                            self.env_params = trial_info_dict['env_params']
                            print 'ASSISTANCE_TYPE', self.env_params['assistance_type']
                            rospy.set_param('assistance_type', self.env_params['assistance_type'])
                            self.set_mode_request = SetModeRequest()
                            self.set_mode_request.mode_index = DIM_TO_MODE_INDEX[self.env_params['start_mode']]
                            status = self.set_mode_srv(self.set_mode_request)
                            self.env.update_params(self.env_params)
                            self.env.reset()
                            self.env.render()
                            self.trial_marker_pub.publish('start')
                            self.trial_index_pub.publish(trial_info_filename_index)
                            self.trial_start_time = time.time()
                            is_done = False
                            self.is_restart = False
                        else:
                            self.shutdown_hook('Reached end of training. Training session timed out. End of session')
                            break

                if self.restart:
                    if not self.training:
                        print("RESTART INITIATED")
                        self.trial_marker_pub.publish('restart')
                        self.restart = False
                        time.sleep(5.0)
                        #TODO should I be incrementing trial index here or should I just restart the same trial?
                        if self.trial_index == len(self.metadata_index):
                            self.shutdown_hook('Reached end of trial list. End of session')
                            break
                        trial_info_filename_index = self.metadata_index[self.trial_index]
                        trial_info_filepath = os.path.join(self.trial_info_dir_path, str(trial_info_filename_index) + '.pkl')
                        assert os.path.exists(trial_info_filepath) is not None
                        with open(trial_info_filepath, 'rb') as fp:
                            trial_info_dict = pickle.load(fp)
                        assert 'env_params' in trial_info_dict
                        trial_info_dict['env_params']['training'] = self.training
                        self.env_params = trial_info_dict['env_params']
                        print 'ASSISTANCE_TYPE', self.env_params['assistance_type']
                        rospy.set_param('assistance_type', self.env_params['assistance_type'])
                        self.set_mode_request = SetModeRequest()
                        self.set_mode_request.mode_index = DIM_TO_MODE_INDEX[self.env_params['start_mode']]
                        status = self.set_mode_srv(self.set_mode_request)
                        self.env.update_params(self.env_params)
                        self.env.reset()
                        self.env.render()
                        self.trial_marker_pub.publish('start')
                        self.trial_index_pub.publish(trial_info_filename_index)
                        self.trial_start_time = time.time()
                        is_done = False
                
                # print(self.input_action)
                robot_continuous_position, robot_continuous_orientation, robot_gripper_angle, robot_linear_velocity, robot_angular_velocity, robot_discrete_state, is_done = self.env.step(self.input_action)

                self.robot_state.header.stamp = rospy.Time.now()
                self.robot_state.robot_continuous_position = robot_continuous_position
                self.robot_state.robot_continuous_orientation = robot_continuous_orientation
                self.robot_state.robot_linear_velocity = robot_linear_velocity
                self.robot_state.robot_angular_velocity = robot_angular_velocity
                self.robot_state.robot_gripper_angle = robot_gripper_angle

                self.robot_state.robot_discrete_state.discrete_location = robot_discrete_state[0]
                self.robot_state.robot_discrete_state.discrete_orientation = robot_discrete_state[1]
                self.robot_state.robot_discrete_state.mode = robot_discrete_state[2]
                self.robot_state.robot_discrete_state.discrete_gripper_angle = robot_discrete_state[3]
                self.robot_state_pub.publish(self.robot_state)
                
                if self.terminate:
                    self.shutdown_hook('Session terminated')
                    break

                self.env.render()
            r.sleep()
    
    def joy_callback(self, msg):
        self.input_action['human'] = msg
    
    def sc_callback(self, msg):
        self.input_action['stabilizing_control'] = msg
    
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
    
if __name__ == '__main__':
    subject_id = sys.argv[1]
    assistance_block = sys.argv[2]
    block_id = sys.argv[3]
    training = int(sys.argv[4])
    print type(subject_id), type(block_id), type(training)
    s = Simulator(subject_id, assistance_block, block_id, training)
    rospy.spin()


