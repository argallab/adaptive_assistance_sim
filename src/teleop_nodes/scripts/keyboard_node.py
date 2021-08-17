#!/usr/bin/env python
# Code developed by Mahdieh Nejati Javaremi*, Deepak Gopinath* in December 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

import rospy
import sys
import numpy as np
import threading
from control_input import ControlInput
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool
from std_msgs.msg import MultiArrayDimension
from teleop_nodes.msg import CartVelCmd
from teleop_nodes.msg import ModeSwitch
from std_msgs.msg import Int16MultiArray
from dynamic_reconfigure.server import Server as DynamicReconfigureServer
from teleop_nodes.cfg import KeyboardModeSwitchParadigmConfig
from teleop_nodes.srv import SetMode, SetModeRequest, SetModeResponse
from std_srvs.srv import SetBool, SetBoolResponse
import curses

npa = np.array


# Code adapted from ros-teleop/teleop_tools/key_teleop.py from PAL Robotics SL (Author Siegfried-A. Gevatter)
class TextWindow:
    _screen = None
    _window = None
    _num_lines = None

    def __init__(self, stdscr, lines=10):
        self._screen = stdscr
        self._screen.nodelay(True)
        curses.curs_set(0)

        self._num_lines = lines

    def read_key(self):
        keycode = self._screen.getch()
        return keycode if keycode != -1 else None

    def clear(self):
        self._screen.clear()

    def write_line(self, lineno, message):
        if lineno < 0 or lineno >= self._num_lines:
            raise ValueError("lineno out of bounds")
        height, width = self._screen.getmaxyx()
        y = int((height / self._num_lines) * lineno)
        x = 10
        for text in message.split("\n"):
            text = text.ljust(width)
            self._screen.addstr(y, x, text)
            y += 1

    def refresh(self):
        self._screen.refresh()

    def beep(self):
        curses.flash()


class KeyTeleop(ControlInput):

    """ data is a CartVel message """

    def __init__(self, interface, robot_dim=6, finger_dim=3):

        # Initialize
        self._interface = interface
        self.robot_dim = robot_dim
        self.finger_dim = finger_dim
        ControlInput.__init__(self, self.robot_dim, self.finger_dim)

        self.initialize_variables()
        self.initialize_publishers()
        self.initialize_services()

        # initial mode message sent to be used by any feedback display or montior
        self.publish_mode()

    def initialize_variables(self):

        self._hz = rospy.get_param("~hz", 100)

        if self.finger_dim > 0:
            self.effective_finger_dim = 1
        else:
            self.effective_finger_dim = 0

        # Initialize Mode Switching Related
        self.modeswitch_msg = ModeSwitch()
        self.mode_msg = Int16MultiArray()

        self._mode = 0
        self._mode_switch_count = 0
        self.mode_switch_paradigm = rospy.get_param("/mode_switch_paradigm", 1)
        self.interface_dim = rospy.get_param("/interface_dim", 1)
        self._last_pressed = {}

        # keys that correspond to motion axes
        self.movement_bindings = {
            curses.KEY_UP: (1, 0),
            curses.KEY_DOWN: (-1, 0),
            curses.KEY_LEFT: (0, 1),
            curses.KEY_RIGHT: (0, -1),
        }
        # text used for feedback corresponding to movement_bindings
        self.motion_binding_text = {0: "up and down arrow", 1: "left and right arrow"}
        # keys that correspond to mode switching (page up and page down)
        self.mode_switch_bindings = {
            curses.KEY_NPAGE: 1,
            curses.KEY_PPAGE: -1,
        }
        # control mode based on control space dimension and interface dimension
        self.mode_binding = self.get_mode_dimension()

        self._linear_rate = rospy.get_param("/jaco_velocity_limits/linear_rate", 10)
        self._angular_rate = rospy.get_param("/jaco_velocity_limits/angular_rate", 10)

        # Set up velocity command and load velocity limits to be sent to hand nodes
        self.gripper_vel_limit = rospy.get_param("/jaco_velocity_limits/gripper_vel_limit", 7)
        # initliaze velocity multiplier to proper length
        self._vel_multiplier = np.ones(self.robot_dim + self.finger_dim) * 1
        # set linear and angular rates
        self._vel_multiplier[0 : self.robot_dim + 1] = self._linear_rate
        self._vel_multiplier[2] = self._angular_rate
        self._max_cart_vel = np.ones(self.robot_dim + self.finger_dim) * rospy.get_param(
            "/jaco_velocity_limits/max_cart_vel_mulitplier", 1
        )
        if self.finger_dim > 0:
            for j in range(self.finger_dim):
                self._max_cart_vel[self.robot_dim + j] = self.gripper_vel_limit
        if not len(self._max_cart_vel) == self.robot_dim + self.finger_dim:
            rospy.logerr("Length of max_cart_vel does not equal number of joints!")
        self._cart_vel = np.zeros(self.robot_dim + self.finger_dim)

        # initialize output velocity message
        self.send_msg = CartVelCmd()
        _dim = [MultiArrayDimension()]
        _dim[0].label = "cartesian_velocity"
        _dim[0].size = self.robot_dim + self.finger_dim
        _dim[0].stride = self.robot_dim + self.finger_dim
        self.send_msg.velocity.layout.dim = _dim
        self.send_msg.velocity.data = np.zeros_like(self._cart_vel)
        self.send_msg.header.stamp = rospy.Time.now()
        self.send_msg.header.frame_id = "keyboard"

    def initialize_publishers(self):
        self.startSend("/user_vel")
        self.mode_switch_pub = rospy.Publisher("/mode_switches", ModeSwitch, queue_size=1)
        self.modepub = rospy.Publisher("/mi/current_mode", Int16MultiArray, queue_size=1)

    def initialize_services(self):
        rospy.Service("/teleop_node/set_mode", SetMode, self.set_mode)
        paradigm_reconfigure_srv = DynamicReconfigureServer(KeyboardModeSwitchParadigmConfig, self.reconfig_paradigm_cb)

    #######################################################################################
    #                           FUNCTIONS FOR SWITCHING MODES                             #
    #######################################################################################

    def get_mode_dimension(self):
        # TO DO(mahdieh): I don't think the current mode method is efficient, I would prefer
        # the mode display code (on arduino side) to look for list of strings instead of the way it is now. Will make
        # code here and there more straight forward and can remove get_digits function.
        # The number of modes is the quotient of robot dimension/interface dimension, plus another dim if there is a remainder or if there is a gripper
        # 1-D and 2-D have been tested and validated
        quotient, remainder = divmod(self.robot_dim, self.interface_dim)
        # if gripper exists, add to the remainder mode (which will be either 0 or 1)
        j = 1
        mode_bindings = {}
        for i in range(quotient):
            mode_axes = []
            for k in range(self.interface_dim):
                mode_axes.append(j)
                j += 1
            mode_bindings[i] = mode_axes
        # figuring out if last mode is two dimensional or only one dimensional
        additional_mode = remainder + self.effective_finger_dim
        if additional_mode == 1:
            mode_bindings[i + 1] = [j]
        elif (
            additional_mode == 2
        ):  # to do: might have to change logic for 3D (I haven't gone through this part of the login with pen and paper)
            mode_axes = []
            for k in range(self.interface_dim):
                mode_axes.append(j)
                j += 1
            mode_bindings[i + 1] = mode_axes
        # number of modes is equal to the quotient and one more if either there is a remainder, gripper, or both
        self.num_modes = quotient + (additional_mode > 0)
        print(mode_bindings)
        return mode_bindings

    # Response to Set Mode Service (eg. if homing)
    def set_mode(self, setmode):
        self._mode = setmode.mode_index
        print("Current mode is ", self._mode, ":", self.mode_binding[self._mode])
        status = SetModeResponse()
        self.publish_mode()
        self.modeswitch_msg.header.frame_id = "service"
        self.publish_modeswitch()
        status = True
        return status

    def publish_mode(self):
        self.modepub.publish(self.mode_msg)

    # publishes mode and updates parameter server
    def publish_modeswitch(self):
        self._mode_switch_count = self._mode_switch_count + 1
        self.modeswitch_msg.header.stamp = rospy.Time.now()
        self.modeswitch_msg.mode = self.mode_binding[self._mode]
        self.modeswitch_msg.num_switches = self._mode_switch_count
        self.mode_switch_pub.publish(self.modeswitch_msg)
        rospy.set_param("mode", self._mode)
        print('Current mode is ', self._mode)

    def reconfig_paradigm_cb(self, config, level):
        self.mode_switch_paradigm = config.mode_switch_paradigm
        self.interface_dim = config.keyboard_dimension
        return config

    # checks whether to switch mode, and changes cyclically
    def handle_mode_switching(self, msg):
        # Paradigm 1: One-way mode switching
        ####################################
        if self.mode_switch_paradigm == 1:
            if int(msg) == 338:  # to do: not working with curses.NPAGE, figure this out
                self._mode = (self._mode + 1) % self.num_modes

        # Paradigm 2: Two-way mode switching
        ####################################
        elif self.mode_switch_paradigm == 2:
            self._mode = (self._mode + self.mode_switch_bindings[msg]) % self.num_modes

        self.mode_msg.data = self.mode_binding[self._mode]
        self.modeswitch_msg.header.frame_id = "user"
        self.publish_modeswitch()
        self.publish_mode()

    #######################################################################################
    #                           FUNCTIONS FOR SETTING VELOCITIES                          #
    #######################################################################################
    def zero_vel(self):
        for i in range(0, self.robot_dim + self.finger_dim):
            self._cart_vel[i] = 0

    def one_mode_zero_vel(self, mode):
        self._cart_vel[mode] = 0

    # def get_digits(self, num):
    #   remaining = num
    #   yield remaining % len

    def handle_velocities(self):
        now = rospy.get_time()
        keys = []
        for a in self._last_pressed:
            if now - self._last_pressed[a] < 0.4:
                keys.append(a)
        dim = np.zeros(self.interface_dim)
        for d in range(len(dim)):
            for k in keys:
                vel = self.movement_bindings[k]
                dim[d] += vel[d]
                # print('dim', dim, 'dim[d]:', dim[d], 'vel:', vel)
        axes = self.mode_binding[self._mode]
        for i in range(len(axes)):
            self._cart_vel[int(axes[i]) - 1] = dim[i] * self._vel_multiplier[i]

    #######################################################################################
    #                                 FUNCTIONS FOR TERMINAL                                      #
    #######################################################################################
    # determines whether keys of interest were pressed
    def _key_pressed(self, keycode):
        if keycode == ord("q"):
            self._running = False
            rospy.signal_shutdown("Bye")
        elif keycode in self.movement_bindings:
            self._last_pressed[keycode] = rospy.get_time()
        elif keycode in self.mode_switch_bindings:
            self.handle_mode_switching(keycode)

    def write_to_terminal(self):
        self._interface.clear()
        axes = self.mode_binding[self._mode]
        self._interface.write_line(2, "mode is %s" % (str(self.mode_binding[self._mode])))
        for i in range(len(axes)):
            self._interface.write_line(3 + i, "Use %s to move in dimension %s" % (self.motion_binding_text[i], axes[i]))
        self._interface.write_line(7, "q to exit")
        self._interface.refresh()

    #######################################################################################
    #                                 MAIN FUNCTIONS                                      #
    #######################################################################################
    def run(self):
        rate = rospy.Rate(self._hz)
        self._running = True
        while self._running:
            while True:
                keycode = self._interface.read_key()
                if keycode is None:
                    break
                self._key_pressed(keycode)
            self.handle_velocities()
            self.handle_threading()
            self.write_to_terminal()
            rate.sleep()

    # handles threading for parallel programming used in receive function
    def handle_threading(self):
        self.send_msg.velocity.data = self._cart_vel
        self.send_msg.header.stamp = rospy.Time.now()
        self.lock.acquire()
        try:
            self.data = self.send_msg
        finally:
            self.lock.release()

    # function required from abstract in control_input
    def getDefaultData(self):
        # since this sends velocity, the default will be all zeros.
        self.lock.acquire()
        try:
            self.data = CartVelCmd()
        finally:
            self.lock.release()


def main(stdscr, robot_dim, finger_dim):
    rospy.init_node("keyboard_node", anonymous=True)
    app = KeyTeleop(TextWindow(stdscr), int(robot_dim), int(finger_dim))
    app.run()


if __name__ == "__main__":
    try:
        curses.wrapper(main, sys.argv[1], sys.argv[2])
    except rospy.ROSInterruptException:
        pass

