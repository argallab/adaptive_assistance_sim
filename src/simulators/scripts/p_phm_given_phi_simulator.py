#!/usr/bin/env python
# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

import rospy
import time
from sensor_msgs.msg import Joy
from std_msgs.msg import String
from simulators.msg import Command
from envs.p_phm_given_phi_env import PPhmGivenPhiEnv
from adaptive_assistance_sim_utils import *
import pyglet
import sys
from random import randrange
import threading

class PPhmGivenPhiSim(object):
    def __init__(self, duration=1.0, iterations=1):
        # initialization
        rospy.init_node("p_um_given_ui_simulator")
        self.initialize_subscribers()
        self.initialize_publishers()

        self.duration = float(duration) # duration command text is displayed on screen
        self.iterations = int(iterations) # number of iterations each command is to be displayed on screen
        self.countdown_duration = 1.5
        self.input_time = rospy.get_rostime()
        self.command_list = []

        self.command_msg = Command()

        env_params = dict()
        env_params['text'] = ''

        self.env = PPhmGivenPhiEnv(env_params)
        self.env.reset()

        self.generate_command_list() # generate random order of commands

        self.event = threading.Event()
        self.lock = threading.Lock()

        self.count = 0

    def initialize_subscribers(self):
        rospy.Subscriber('/keyboard_entry', String, self.keyboard_callback)
        rospy.Subscriber('/joy_sip_puff', Joy, self.joy_callback)

    def initialize_publishers(self):
        # for ros bag purposes (not being used for any code logic)
        self.command_pub = rospy.Publisher('command_prompt', Command, queue_size=1)

    def publish_command(self, msg):
        self.command_msg.header.stamp = rospy.Time.now()
        self.command_msg.command = msg
        self.command_pub.publish(self.command_msg)

    # start experiment
    def keyboard_callback(self, msg):
        # Start experiment
        if msg.data == 's':
            for i in range(len(EXPERIMENT_START_COUNTDOWN)):
                self.call_render(EXPERIMENT_START_COUNTDOWN[i], self.countdown_duration)
            self.command_following_task()

    def joy_callback(self, msg):
        self.input_time = msg.header.stamp

    # randomize commands iterations
    def generate_command_list(self):
        for i in range(self.iterations):
            commands = LOW_LEVEL_COMMANDS[:]
            for j in range(len(commands)):
                rand_index = randrange(len(commands))
                self.command_list.append(commands[rand_index])
                commands.pop(rand_index)
        print 'total commands: ', (len(self.command_list))

    # display commands for desired duration and (wait for user to stop input before sending next command)
    def command_following_task(self):
        i = 0
        while i < len(self.command_list):
            self.command_time = rospy.get_rostime()
            self.publish_command(self.command_list[i])
            self.call_render(self.command_list[i], self.duration)
            self.publish_command('')
            self.call_render('', self.countdown_duration)
            if (self.input_time < self.command_time):
                self.command_list.append(self.command_list[i])
            i += 1
            print(i, len(self.command_list))
        self.call_render('ALL DONE! :D', self.duration)
        self.env.viewer.close()

    # set new text message and render
    def call_render(self, msg, duration):
        self.env.env_params['text'] = msg
        self.env.reset()
        self.env.render()
        rospy.sleep(duration)

    def shutdown_hook(self):
        pass

if __name__ == '__main__':
    PPhmGivenPhiSim(sys.argv[1], sys.argv[2])
    rospy.spin()
