#!/usr/bin/env python
# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

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
from std_msgs.msg import Int16
from teleop_nodes.cfg import SipPuffModeSwitchParadigmConfig
from teleop_nodes.srv import SetMode, SetModeRequest, SetModeResponse
from std_srvs.srv import SetBool, SetBoolResponse
npa = np.array



class SNPInput(ControlInput):

  ''' data is a CartVel message '''
  def __init__(self, robot_dim = 6, finger_dim=3):

    # Initialize
    ControlInput.__init__(self, robot_dim, finger_dim)

    self.initialize_subscribers()
    self.initialize_publishers()
    self.initialize_services()

    # Initialize
    self.modeswitch_msg = ModeSwitch()
    self.mode_msg = Int16()
    self._mode = 0
    self._mode_switch_count = 0
    self._switch_direction = 1
    self._button_latch_time = 0.8
    self._positive_latch = 0
    self._negative_latch = 0
    self._latch_lock = 0
    self._latch_start_time = rospy.get_time()
    self._latch_command_duration = 0.5 #seconds
    self._lock_input = True # to prevent constant mode switching
    self.mode_switch_paradigm = 2 #two-way mode switching
    self.motion_paradigm = 3 #constant velocity paradigm

    # Set up velocity command and load velocity limits to be sent to hand nodes
    if self.finger_dim > 0:
        self.effective_finger_dim = 1 #a single dimension's velocity is mapped to ALL the fingers of the robot
    else:
        self.effective_finger_dim = 0
    self.velocity_scale = rospy.get_param("/snp_velocity_scale") #currently scalar. Could be vector for individual dimension scale
    self.gripper_vel_limit = rospy.get_param("/jaco_velocity_limits/gripper_vel_limit")
    self._vel_multiplier = self.velocity_scale*np.ones(self.robot_dim + self.finger_dim)*1
    ## TODO (deepak). Get different multipler for rotation and assigned indices for rotation dimensions:
    self._vel_multiplier[2] = 0.4*self._vel_multiplier[2]
    self._max_cart_vel = np.ones(self.robot_dim + self.finger_dim)*rospy.get_param("/jaco_velocity_limits/max_cart_vel_mulitplier")
    if self.finger_dim > 0:
        for j in range(finger_dim):
            self._max_cart_vel[self.robot_dim + j] = self.gripper_vel_limit
    if not len(self._max_cart_vel) == self.robot_dim + self.finger_dim:
        rospy.logerr('Length of max_cart_vel does not equal number of joints!')
    self._cart_vel = np.zeros(self.robot_dim + self.finger_dim)

    # initialize output velocity message
    self.send_msg = CartVelCmd()
    _dim = [MultiArrayDimension()]
    _dim[0].label = 'cartesian_velocity'
    _dim[0].size = self.robot_dim + self.finger_dim
    _dim[0].stride = self.robot_dim + self.finger_dim
    self.send_msg.velocity.layout.dim = _dim
    self.send_msg.velocity.data = np.zeros_like(self._cart_vel)
    self.send_msg.header.stamp = rospy.Time.now()
    self.send_msg.header.frame_id = 'snp'

    self.publish_mode()

  def initialize_subscribers(self):
    rospy.Subscriber('/joy_sip_puff', Joy, self.receive)
    rospy.Subscriber('/chin_button', Bool, self.switchMode) # only for chin button mode switching

  def initialize_publishers(self):
    self.startSend('/user_vel')
    self.mode_switch_pub = rospy.Publisher('/mode_switches', ModeSwitch, queue_size=1)
    self.modepub = rospy.Publisher("/mi/current_mode", Int16, queue_size=1)

  def initialize_services(self):
    rospy.Service('/teleop_node/set_mode', SetMode, self.set_mode)
    rospy.Service('/teleop_node/stop_latch', SetBool, self.stop_latch)

  #######################################################################################
  #                           FUNCTIONS FOR SWITCHING MODES                             #
  #######################################################################################

  # Response to Set Mode Service (eg. if homing)
  def set_mode(self, setmode):
    # 0: X, 1: Y, 2: Z, 3: Roll, 4: Pitch, 5: Yaw, 6: Gripper
    self._mode = setmode.mode_index
    print "Current mode is ", self._mode
    status = SetModeResponse()
    self.publish_mode()
    self.modeswitch_msg.header.frame_id = 'service'
    self.publish_modeswitch()
    status = True
    return status

  # TODO: remove this, maybe have arduino listen to the other message and just add one to it
  def publish_mode(self):
    self.modepub.publish(self._mode+1) # +1 for arduino

  # publishes mode and updates parameter server
  def publish_modeswitch(self):
    self._mode_switch_count = self._mode_switch_count+1
    self.modeswitch_msg.header.stamp = rospy.Time.now()
    self.modeswitch_msg.mode = self._mode
    self.modeswitch_msg.num_switches = self._mode_switch_count
    self.mode_switch_pub.publish(self.modeswitch_msg)
    rospy.set_param('mode',self._mode) #update the param server so that the update function in the env can read the current mode. Needed to determine what dimensions need to be zeroed out
    print "Num of mode switches %d" % self._mode_switch_count

  # checks whether to switch mode, and changes cyclically
  def switchMode(self, msg):
    switch = False
    # Paradigm 1: One-way mode switching
    ####################################
    if self.mode_switch_paradigm == 1:
      if msg.buttons[0] or msg.buttons[3]:
        self._mode = (self._mode + 1) % self.robot_dim + self.effective_finger_dim
        switch = True

    # Paradigm 2: Two-way mode switching
    ####################################
    elif self.mode_switch_paradigm == 2:
      if msg.buttons[0]:
        self._mode = (self._mode + 1) % self.robot_dim + self.effective_finger_dim
        switch = True
      elif msg.buttons[3]:
        self._mode = (self._mode - 1) % self.robot_dim + self.effective_finger_dim
        switch = True

    # Paradigm 3: Chin button mode switching
    # #####################################
    elif self.mode_switch_paradigm == 3:
      if button_msg.data == True:
        self._mode = (self._mode + self._switch_direction) % self.robot_dim + self.effective_finger_dim
        switch = True
        self._old_time = rospy.get_time()
      elif rospy.get_time() - self._old_time > self._button_latch_time:
        self._switch_direction *= -1
        print "Changed direction "

    if switch:
      self._lock_input = True
      print "************MODE IS NOW ", self._mode, " *******************"
      self.modeswitch_msg.header.frame_id = 'user'
      self.publish_modeswitch()
      self.publish_mode()



  #######################################################################################
  #                           FUNCTIONS FOR SETTING VELOCITIES                          #
  #######################################################################################
  def zero_vel(self):
    for i in range(0,self.robot_dim + self.finger_dim):
        self._cart_vel[i] = 0

  def one_mode_zero_vel(self, mode):
    self._cart_vel[mode] = 0

  def check_latch_time(self):
    if (rospy.get_time() - self._latch_start_time) > self._latch_command_duration:
      self._latch_lock = 0
      self.zero_vel()

  def stop_latch(self, booler):
    self._latch_lock = 0
    self.zero_vel()
    status = SetBoolResponse()
    return status

  def handle_velocities(self, msg):
    # # Paradigm 1: Timed-latch
    # ####################################
    if self.motion_paradigm == 1:
      # self.check_latch_time()
      if not self._latch_lock:
        if msg.buttons[1]: # soft puff (+ve motion)
          # latch forward
          self._cart_vel[self._mode] = self._max_cart_vel[self._mode] * self._vel_multiplier[self._mode]
          self._latch_start_time = rospy.get_time()
          self._latch_lock = 1
        elif msg.buttons[2]: # soft sip (-ve motion)
          # latch reverse
          self._cart_vel[self._mode] = -self._max_cart_vel[self._mode] * self._vel_multiplier[self._mode]
          self._latch_start_time = rospy.get_time()
          self._latch_lock = 1

    # # Paradigm 2: Self-stop latch
    # ####################################
    # TODO: May need to zero out velocities during mode switching
    if self.motion_paradigm == 2:
      if self._latch_lock:
        if msg.buttons[1]==0 and msg.buttons[2]==0 and msg.buttons[0]==0 and msg.buttons[3]==0:
          self._latch_lock = 0
      elif msg.buttons[1]: # soft puff (+ve motion)
        if self._negative_latch: # stop reverse latch
          self.zero_vel()
          self._negative_latch = 0
          self._latch_lock = 1
        else: # latch forward
          self._cart_vel[self._mode] = self._max_cart_vel[self._mode] * self._vel_multiplier[self._mode]
          self._positive_latch = 1
          self._latch_lock = 1
      elif msg.buttons[2]: # soft sip (-ve motion)
        if self._positive_latch: # stop forward latch
          self.zero_vel()
          self._positive_latch = 0
          self._latch_lock = 1
        else: # latch reverse
          self._cart_vel[self._mode] = -self._max_cart_vel[self._mode] * self._vel_multiplier[self._mode]
          self._negative_latch = 1
          self._latch_lock = 1
      # this zeros out when mode switching is happening. If you want to turn and move, remove this

    # Paradigm 3: Constant velocity
    ####################################
    if self.motion_paradigm == 3:
      if msg.buttons[1] :
        self._cart_vel[self._mode] = self._max_cart_vel[self._mode] * self._vel_multiplier[self._mode]
      elif msg.buttons[2]:
        self._cart_vel[self._mode] = -self._max_cart_vel[self._mode] * self._vel_multiplier[self._mode]
      else:
        self.zero_vel()

    # Paradigm 4: Proportional velocity
    ####################################
    if self.motion_paradigm == 4:
      if msg.buttons[1] or msg.buttons[2]:
        self._cart_vel[self._mode] = -self._max_cart_vel[self._mode] * msg.axes[0] * self._vel_multiplier[self._mode]
      else:
        self.zero_vel()
    # TODO: send zero velocity when mode switch

  # determines whether mode switch should happen, if not, moves robot arm based on mode
  def handle_paradigms(self, msg):
    if self._positive_latch == 0 and self._negative_latch == 0:
      self.switchMode(msg)
    self.handle_velocities(msg)

    # send the velocities to robot
    self.send_msg.velocity.data = self._cart_vel
    self.send_msg.header.stamp = rospy.Time.now()

#######################################################################################
#                                 MAIN FUNCTIONS                                      #
#######################################################################################
  # handles threading for parallel programming used in receive function
  def handle_threading(self):
    self.lock.acquire()
    try:
      self.data = self.send_msg
    finally:
      self.lock.release()

  def latch_timing_threader(self):
    while True:
      self.check_latch_time

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
      self.data = CartVelCmd()
    finally:
      self.lock.release()

if __name__ == '__main__':
  rospy.init_node('sip_puff_node', anonymous=True)
  robot_dim = int(sys.argv[1])
  finger_dim = int(sys.argv[2])
  snp = SNPInput(robot_dim=robot_dim, finger_dim=finger_dim) #this robot dim has to be a parameter. Need to be changed for the 4D robot
  rospy.spin()
