#!/usr/bin/env python
# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

import rospy
import rospkg
import os
from simulators.msg import Command
from envs.p_ui_given_a_env import PUiGivenAEnv
import sys
from random import randrange
from pyglet.window import key

class PUiGivenASim(object):
    def __init__(self, iterations=1, blocks=1):

        # initialization
        rospy.init_node("p_ui_given_a_simulator")
        rospy.on_shutdown(self.shutdown_hook)
        self.iterations = int(iterations)
        self.blocks = int(blocks)

        self.action_list = []

        # TODO: change to input argumen using argparse
        file_dir = os.path.join(rospkg.RosPack().get_path('envs'), 'sprites', 'actions')
        self.actions= ['clockwise', 'counterclockwise', 'down', 'up', 'left', 'right',
                        'mode_switch_right_1', 'mode_switch_right_2', 'mode_switch_right_3',
                        'mode_switch_left_1', 'mode_switch_left_2', 'mode_switch_left_3']

        self.training = ['hard_puff_training','hard_sip_training','soft_puff_training','soft_sip_training']

        env_params = dict()
        env_params['file_dir'] = file_dir
        env_params['img_prompt'] = ''
        env_params['training_prompts'] = self.training[:]
        env_params['actions'] = {}
        env_params['actions']['motion_actions'] = self.actions[:6]
        env_params['actions']['mode_actions'] = self.actions[6:]
        env_params['blocks'] = blocks

        self.env = PUiGivenAEnv(env_params)

        # TODO: cleanup publishers
        self.env.initialize_publishers('action_prompt')
        self.initialize_publishers()
        self.key_input_msg = Command()
        self.user_response_msg = Command()
        self.action_msg = Command()

        self.generate_action_list()
        r = rospy.Rate(100)
        self.env.initialize_viewer()
        self.env.viewer.window.on_key_press = self.key_press_callback

        while not rospy.is_shutdown():
            b, msg = self.env.step()
            if b:
                self.publish_action(msg)
            self.env.render()
            r.sleep()

    # randomize actions
    def generate_action_list(self):
        for i in range(self.iterations):
            actions = self.actions[:]
            for j in range(len(actions)):
                rand_ind = randrange(len(actions))
                self.action_list.append(actions[rand_ind])
                actions.pop(rand_ind)
        self.env.env_params['action_prompts'] = self.action_list[:]
        self.env.reset()

    def initialize_publishers(self):
        self.user_input_pub = rospy.Publisher('keyboard_entry', Command, queue_size = 1)
        self.user_response_pub = rospy.Publisher('user_response', Command, queue_size = 1)
        self.action_pub = rospy.Publisher('action_prompt', Command, queue_size=1)

    def publish_keyboard_input(self, msg):
        self.key_input_msg.header.stamp = rospy.Time.now()
        self.key_input_msg.command = msg
        self.user_input_pub.publish(self.key_input_msg)

    def publish_user_response(self, msg):
        self.user_response_msg.header.stamp = rospy.Time.now()
        self.user_response_msg.command = msg
        self.user_response_pub.publish(self.user_response_msg)

    def publish_action(self, msg):
        self.action_msg.header.stamp = rospy.Time.now()
        self.action_msg.command = msg
        self.action_pub.publish(self.action_msg)

    def key_press_callback(self, k, mode):
        self.publish_keyboard_input(str(k))
        if k==key.S:
            self.env.env_params['start_prompt'] = True
            self.publish_keyboard_input('s')
            self.env.reset()
        if k==key._1:
            self.env.env_params['next_prompt'] = True
            b = self.env._get_user_input() # if user was allowed to give a response (i.e. during prompt)
            if b:
                self.publish_user_response('1')
        if k==key._2:
            self.env.env_params['next_prompt'] = True
            b = self.env._get_user_input()
            if b:
                self.publish_user_response('2')
        if k==key._3:
            self.env.env_params['next_prompt'] = True
            b = self.env._get_user_input()
            if b:
                self.publish_user_response('3')
        if k==key._4:
            self.env.env_params['next_prompt'] = True
            b = self.env._get_user_input()
            if b:
                self.publish_user_response('4')
        if k==key.LEFT:
            self.env.env_params['back'] = True
            self.env._get_user_input()
        if k==key.RIGHT:
            self.env.env_params['next'] = True
            self.env._get_user_input()

    def shutdown_hook(self):
        pass

if __name__ == '__main__':
    PUiGivenASim(sys.argv[1], sys.argv[2])
    rospy.spin()
