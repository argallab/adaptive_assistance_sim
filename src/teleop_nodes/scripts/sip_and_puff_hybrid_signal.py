#!/usr/bin/env python
# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

import rospy
import sys
import numpy as np
import threading
from hybrid_control_input import HybridControlInput
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool
from std_msgs.msg import MultiArrayDimension
from teleop_nodes.msg import InterfaceSignal
from teleop_nodes.msg import ModeSwitch
from std_msgs.msg import Int16, String
from teleop_nodes.cfg import SipPuffModeSwitchParadigmConfig
from teleop_nodes.srv import SetMode, SetModeRequest, SetModeResponse
from std_srvs.srv import SetBool, SetBoolResponse
from envs.srv import SwitchModeSrv, SwitchModeSrvRequest, SwitchModeSrvResponse

npa = np.array


class SNPInput(HybridControlInput):
    def __init__(self, interface_velocity_dim=1):
        # Initialize
        HybridControlInput.__init__(self, interface_velocity_dim)

        self.initialize_subscribers()
        self.initialize_publishers()

        self.mode_switch_paradigm = 2  # two-way mode switching
        self.motion_paradigm = 3  # constant velocity paradigm
        self._lock_input = True  # to prevent constant mode switching

        self.velocity_scale = rospy.get_param("/snp_velocity_scale")
        self._vel_multiplier = self.velocity_scale * np.ones(self.interface_velocity_dim) * 1

        self.send_msg = InterfaceSignal()
        self.send_msg.interface_signal = [0] * self.interface_velocity_dim

        rospy.loginfo("Waiting for sim_env node ")
        rospy.wait_for_service("/sim_env/switch_mode_in_robot")
        rospy.loginfo("Found sim_env")

        self.switch_mode_in_robot_service = rospy.ServiceProxy("/sim_env/switch_mode_in_robot", SwitchModeSrv)
        self.switch_mode_pub = rospy.Publisher("mode_switch", String, queue_size=1)

    def initialize_subscribers(self):
        rospy.Subscriber("/joy_sip_puff", Joy, self.receive)

    def initialize_publishers(self):
        self.startSend("/user_vel")

    def _handle_mode_switch_action(self, msg):
        switch = False
        req = SwitchModeSrvRequest()
        if self.mode_switch_paradigm == 1:
            if msg.buttons[0] or msg.buttons[3]:
                req.mode_switch_action = "to_mode_r"
                self.switch_mode_in_robot_service(req)
                self.switch_mode_pub.publish("to_mode_r")
                switch = True

        elif self.mode_switch_paradigm == 2:
            if msg.buttons[0]:
                req.mode_switch_action = "to_mode_r"
                self.switch_mode_in_robot_service(req)
                self.switch_mode_pub.publish("to_mode_r")
                switch = True
            elif msg.buttons[3]:
                req.mode_switch_action = "to_mode_l"
                self.switch_mode_in_robot_service(req)
                self.switch_mode_pub.publish("to_mode_l")
                switch = True

        if switch:
            self._lock_input = True

        return switch

    def _handle_velocity_action(self, msg):
        if self.motion_paradigm == 3:  # fixed velocity
            if msg.buttons[1]:  # soft puff
                # add vel multiplication element wise mult with self._vel_multiplier
                self.send_msg.interface_signal = [+1 * self.velocity_scale]
            elif msg.buttons[2]:  # soft sip
                self.send_msg.interface_signal = [-1 * self.velocity_scale]
            else:
                self.send_msg.interface_signal = [0]

    def handle_paradigms(self, msg):
        switch = self._handle_mode_switch_action(msg)
        self._handle_velocity_action(msg)

        self.send_msg.header.stamp = rospy.Time.now()
        self.send_msg.mode_switch = switch

    def handle_threading(self):
        self.lock.acquire()
        try:
            self.data = self.send_msg
        finally:
            self.lock.release()

    # the main function, determines velocities to send to robot
    def receive(self, msg):
        if msg.header.frame_id == "input stopped":
            self._lock_input = False
        if not self._lock_input:
            self.handle_paradigms(msg)
            self.handle_threading()

    # function required from abstract in control_input
    def getDefaultData(self):
        # since this sends velocity, the default will be all zeros.
        self.lock.acquire()
        try:
            self.data = InterfaceSignal()
        finally:
            self.lock.release()


if __name__ == "__main__":
    rospy.init_node("sip_puff_interface_signal_node", anonymous=True)
    interface_velocity_dim = int(sys.argv[1])
    snp = SNPInput(interface_velocity_dim=interface_velocity_dim)
    rospy.spin()
