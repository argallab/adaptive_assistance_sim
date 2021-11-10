#!/usr/bin/env python
# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

import rospy
import numpy as np
from sensor_msgs.msg import Joy
import time
import collections
from std_srvs.srv import Empty, EmptyRequest, EmptyResponse
from teleop_nodes.srv import GoalInferenceInfo, GoalInferenceInfoRequest, GoalInferenceInfoResponse
import sys
import os
import rospkg
import threading

sys.path.append(os.path.join(rospkg.RosPack().get_path("simulators"), "scripts"))
# from adaptive_assistance_sim_utils import AssistanceType

npa = np.array

"""Reads raw sip_and_puff joy data, does thresholding and buffering
 and returns data as hard puff, soft puff, soft sip, hard sip.
 Input and output are both /sensor_msgs/joy"""


class SNPMapping(object):

    """data is a Float32Array message"""

    def __init__(self, training=False):

        # Initialize
        rospy.init_node("sip_puff_mapping", anonymous=True)
        # Sip/Puff Control Thresholds
        # sip values are positive, puff values are negative in the axes
        # Note that for puff the actual values are negative. The "lower" and "higher" make sense only if absolute values are considered
        # creating limits prevents unintended soft sips and puffs, and creates deadzone between upper and latch limit
        self.lower_sip_limit = rospy.get_param("/sip_and_puff_thresholds/lower_sip_limit")
        self.lower_puff_limit = rospy.get_param("/sip_and_puff_thresholds/lower_puff_limit")
        self.soft_sip_max_limit = rospy.get_param("/sip_and_puff_thresholds/soft_sip_max_limit")
        self.soft_puff_max_limit = rospy.get_param("/sip_and_puff_thresholds/soft_puff_max_limit")
        self.hard_sip_min_limit = rospy.get_param("/sip_and_puff_thresholds/hard_sip_min_limit")
        self.hard_puff_min_limit = rospy.get_param("/sip_and_puff_thresholds/hard_puff_min_limit")
        self.hard_puff_max_limit = rospy.get_param("/sip_and_puff_thresholds/hard_puff_max_limit")
        self.hard_sip_max_limit = rospy.get_param("/sip_and_puff_thresholds/hard_sip_max_limit")

        # latch limit governs the threshold for direction and main mode switches
        self._ignore_input_counter = 0
        self._num_inputs_to_ignore = 10
        self._button_latch_time = 0.8
        self._old_time = rospy.get_time()
        self.old_msg = Joy()
        self.old_msg.axes = np.zeros(1)
        self.command_to_button_index_map = collections.OrderedDict(
            {"Hard Puff": 0, "Soft Puff": 1, "Soft Sip": 2, "Hard Sip": 3}
        )

        self.raw_joy_message = Joy()
        self.raw_joy_message.axes = np.zeros(2)
        self.raw_joy_message.buttons = np.zeros(8)

        # Initialize publisher and subscribers
        rospy.Subscriber("/joy", Joy, self.joy_callback)
        self.before_inference_pub = rospy.Publisher("joy_sip_puff_before", Joy, queue_size=1)
        self.after_inference_pub = rospy.Publisher("joy_sip_puff", Joy, queue_size=1)

        # Published velocity message
        self.send_msg = Joy()
        self.send_msg.header.stamp = rospy.Time.now()
        self.send_msg.header.frame_id = "Zero Band"
        self.send_msg.axes = np.zeros(1)  # pressure ([-1, 1])
        self.send_msg.buttons = np.zeros(4)  # hard puff, soft puff, soft sip, hard sip

        self.before_send_msg = Joy()
        self.before_send_msg.header.stamp = rospy.Time.now()
        self.before_send_msg.header.frame_id = "Zero Band"
        self.before_send_msg.axes = np.zeros(1)  # pressure ([-1, 1])
        self.before_send_msg.buttons = np.zeros(4)  # hard puff, soft puff, soft sip, hard sip
        self.training = training

        # if not self.training:
        rospy.loginfo("Waiting for goal_inference node ")
        rospy.wait_for_service("/goal_inference/handle_inference")
        rospy.loginfo("Found goal_inference")
        self.goal_inference_service = rospy.ServiceProxy("/goal_inference/handle_inference", GoalInferenceInfo)

        self.running = False
        self.runningCV = threading.Condition()
        self.rate = rospy.Rate(10)
        rospy.Service("/snp_mapping/start_thread", Empty, self.start_thread)

    def start_thread(self, req):
        print("start running")
        response = EmptyResponse()
        self.running = True
        print(self.running)
        return response

    # #######################################################################################
    #                           Check Sip and Puff Limits                                   #
    #######################################################################################
    # checks whether within limits, otherwise air velocity in dead zone (essentailly zero)
    # written this way to make debugging easier if needed
    # labels hard and soft sips and puffs, buffers out
    # def update_assistance_type(self):
    #     # TODO maybe replace with service
    #     self.assistance_type = rospy.get_param("assistance_type", 2)
    #     if self.assistance_type == 0:
    #         self.assistance_type = AssistanceType.Filter
    #     elif self.assistance_type == 1:
    #         self.assistance_type = AssistanceType.Corrective
    #     elif self.assistance_type == 2:
    #         self.assistance_type = AssistanceType.No_Assistance

    def checkLimits(self, airVelocity):
        if self.lower_puff_limit < airVelocity < self.lower_sip_limit:
            self.send_msg.header.frame_id = "None"
            self.send_msg.buttons = np.zeros(4)
        elif self.lower_sip_limit <= airVelocity <= self.soft_sip_max_limit:  # register as soft sip
            self.send_msg.header.frame_id = "Soft Sip"
            self.send_msg.buttons[2] = 1
        elif self.soft_puff_max_limit <= airVelocity <= self.lower_puff_limit:  # register as soft puff
            self.send_msg.header.frame_id = "Soft Puff"
            self.send_msg.buttons[1] = 1
        elif self.hard_puff_max_limit <= airVelocity < self.hard_puff_min_limit:  # register as hard puff
            self.send_msg.header.frame_id = "Hard Puff"
            self.send_msg.buttons[0] = 1
        elif self.hard_sip_min_limit < airVelocity <= self.hard_sip_max_limit:  # register as hard sip
            self.send_msg.header.frame_id = "Hard Sip"
            self.send_msg.buttons[3] = 1
        else:
            if airVelocity < 0:
                self.send_msg.header.frame_id = "None"
            else:
                self.send_msg.header.frame_id = "None"
            self.send_msg.buttons = np.zeros(4)

        self.before_send_msg.header.frame_id = self.send_msg.header.frame_id
        self.before_send_msg.buttons = self.send_msg.buttons
        self.before_inference_pub.publish(self.before_send_msg)

        # self.update_assistance_type()
        # if not self.training:
        request = GoalInferenceInfoRequest()
        request.phm = self.send_msg.header.frame_id
        response = self.goal_inference_service(request)

        self.send_msg.buttons = np.zeros(4)
        if self.send_msg.header.frame_id != "None":
            self.send_msg.buttons[self.command_to_button_index_map[self.send_msg.header.frame_id]] = 1

        return 0

    #######################################################################################
    #                                Raw Joy Callback                                     #
    #######################################################################################
    # recieves raw input, checks for buildup and

    def joy_callback(self, msg):
        self.raw_joy_message = msg

    def step(self):
        # Ignore the leadup to powerful blow that leads to mode switch (ONLY FOR SIP-PUFF SYSTEM, otherwise delete)
        # seems like thread issue if the number to ignore is too high
        # print(self.raw_joy_message)
        if self._ignore_input_counter < self._num_inputs_to_ignore:
            self._ignore_input_counter += 1

        self.send_msg.header.stamp = rospy.Time.now()
        # the pressure level are directly copied from the input joy message.
        self.send_msg.axes[0] = self.raw_joy_message.axes[1]
        self.checkLimits(self.raw_joy_message.axes[1])
        self.after_inference_pub.publish(self.send_msg)

        # prevent robot arm moving after done blowing, zero out velocities
        # the last input in each blow is 0 for buttons
        if self.raw_joy_message.buttons[0] is 0 and self.raw_joy_message.buttons[1] is 0:
            self._ignore_input_counter = 0  # the constraints get
            self.send_msg.header.frame_id = "input stopped"
            self.send_msg.buttons = np.zeros(4)
            self.after_inference_pub.publish(self.send_msg)

        self.old_msg = self.raw_joy_message

    def spin(self):
        rospy.loginfo("Running")
        try:
            while not rospy.is_shutdown():
                self.runningCV.acquire()
                if self.running:
                    self.step()
                    self.rate.sleep()
                else:
                    self.runningCV.wait(1.0)
                self.runningCV.release()
        except KeyboardInterrupt:
            rospy.logdebug("Keyboard interrupt, shutting down")
            rospy.core.signal_shutdown("Keyboard interrupt")


if __name__ == "__main__":

    s = SNPMapping(sys.argv[1])
    s.spin()
