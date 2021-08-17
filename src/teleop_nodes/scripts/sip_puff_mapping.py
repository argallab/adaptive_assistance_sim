#!/usr/bin/env python
# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

import rospy
import numpy as np
from sensor_msgs.msg import Joy
import time
import collections
from dynamic_reconfigure.server import Server
from teleop_nodes.srv import InferCorrect, InferCorrectRequest, InferCorrectResponse
from teleop_nodes.srv import GoalInferModify, GoalInferModifyRequest, GoalInferModifyResponse
import sys
import os
import rospkg

sys.path.append(os.path.join(rospkg.RosPack().get_path("simulators"), "scripts"))
from corrective_mode_switch_utils import AssistanceType

npa = np.array

"""Reads raw sip_and_puff joy data, does thresholding and buffering
 and returns data as hard puff, soft puff, soft sip, hard sip.
 Input and output are both /sensor_msgs/joy"""


class SNPMapping(object):

    """ data is a Float32Array message """

    def __init__(self):

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

        self.assistance_type = rospy.get_param("/assistance_type", 2)
        if self.assistance_type == 0:
            self.assistance_type = AssistanceType.Filter
        elif self.assistance_type == 1:
            self.assistance_type = AssistanceType.Corrective
        elif self.assistance_type == 2:
            self.assistance_type = AssistanceType.No_Assistance

        rospy.loginfo("Waiting for goal_inference_and_correction node ")
        rospy.wait_for_service("/goal_inference_and_correction/handle_inference_and_unintended_actions")
        rospy.loginfo("Found goal_inference_and_correction")

        self.infer_and_correct_service = rospy.ServiceProxy(
            "/goal_inference_and_correction/handle_inference_and_unintended_actions", GoalInferModify
        )

    # #######################################################################################
    #                           Check Sip and Puff Limits                                   #
    #######################################################################################
    # checks whether within limits, otherwise air velocity in dead zone (essentailly zero)
    # written this way to make debugging easier if needed
    # labels hard and soft sips and puffs, buffers out
    def update_assistance_type(self):
        # TODO maybe replace with service
        self.assistance_type = rospy.get_param("assistance_type", 2)
        if self.assistance_type == 0:
            self.assistance_type = AssistanceType.Filter
        elif self.assistance_type == 1:
            self.assistance_type = AssistanceType.Corrective
        elif self.assistance_type == 2:
            self.assistance_type = AssistanceType.No_Assistance

    def checkLimits(self, airVelocity):
        if self.lower_puff_limit < airVelocity < self.lower_sip_limit:
            self.send_msg.header.frame_id = "Zero Band"
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
                self.send_msg.header.frame_id = "Soft-Hard Puff Deadband"
            else:
                self.send_msg.header.frame_id = "Soft-Hard Sip Deadband"
            self.send_msg.buttons = np.zeros(4)

        self.before_send_msg.header.frame_id = self.send_msg.header.frame_id
        self.before_send_msg.buttons = self.send_msg.buttons
        self.before_inference_pub.publish(self.before_send_msg)

        self.update_assistance_type()
        if (
            self.assistance_type != AssistanceType.No_Assistance
            and self.send_msg.header.frame_id != "Zero Band"
            and self.send_msg.header.frame_id != "Soft-Hard Puff Deadband"
            and self.send_msg.header.frame_id != "Soft-Hard Sip Deadband"
            and self.send_msg.header.frame_id != "Input Stopped"
        ):
            request = GoalInferModifyRequest()
            request.phm = self.send_msg.header.frame_id
            response = self.infer_and_correct_service(request)
            if response.ph_modified == "None":
                self.send_msg.header.frame_id = "Zero Band"
            else:
                self.send_msg.header.frame_id = response.ph_modified
            self.send_msg.buttons = np.zeros(4)
            if self.send_msg.header.frame_id != "Zero Band":
                self.send_msg.buttons[self.command_to_button_index_map[self.send_msg.header.frame_id]] = 1

        return 0

    #######################################################################################
    #                                Raw Joy Callback                                     #
    #######################################################################################
    # recieves raw input, checks for buildup and

    def joy_callback(self, msg):
        # Ignore the leadup to powerful blow that leads to mode switch (ONLY FOR SIP-PUFF SYSTEM, otherwise delete)
        # seems like thread issue if the number to ignore is too high
        if (self._ignore_input_counter < self._num_inputs_to_ignore):  
            self._ignore_input_counter += 1

        self.send_msg.header.stamp = rospy.Time.now()
        self.send_msg.axes[0] = msg.axes[1]
        self.checkLimits(msg.axes[1])
        self.after_inference_pub.publish(self.send_msg)

        # prevent robot arm moving after done blowing, zero out velocities
        if msg.buttons[0] is 0 and msg.buttons[1] is 0:  # the last input in each blow is 0 for buttons
            self._ignore_input_counter = 0  # the constraints get
            self.send_msg.header.frame_id = "input stopped"
            self.send_msg.buttons = np.zeros(4)
            self.after_inference_pub.publish(self.send_msg)

        self.old_msg = msg


if __name__ == "__main__":

    SNPMapping()
    rospy.spin()
