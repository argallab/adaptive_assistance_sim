#!/usr/bin/env python
import rospy
import numpy as np
from control_input import ControlInput
from sensor_msgs.msg import Joy
from std_msgs.msg import MultiArrayDimension
from std_msgs.msg import Int16
import tf
import tf.transformations as tfs
from jaco_teleop.msg import CartVelCmd
from jaco_teleop.msg import ModeSwitch
from dynamic_reconfigure.server import Server
from jaco_teleop.cfg import ModeSwitchParadigmConfig
from jaco_teleop.srv import SetMode, SetModeRequest, SetModeResponse


class JoystickInput(ControlInput):
  ''' data is a Float32Array message '''

  def __init__(self):
    ControlInput.__init__(self)

    # variables
    self.mode_msg = Int16()
    self.modeswitch_msg = ModeSwitch()

    # Initialize
    self._mode = 0 # see get_mode_switch_paradigm function for keys
    self.gripper_vel_limit = 3000
    self.waiting_for_release = False
    self.modeSwitchCount = 0

    # load velocity limits
    if rospy.has_param('max_cart_vel'):
      self._max_cart_vel = np.array(rospy.get_param('max_cart_vel'))
    else:
      self._max_cart_vel = np.ones(9)
      self._max_cart_vel[6] = self.gripper_vel_limit
      self._max_cart_vel[7] = self.gripper_vel_limit
      self._max_cart_vel[8] = self.gripper_vel_limit
      rospy.logwarn('No rosparam for max_cart_vel found...Defaulting to max linear velocity of 50 cm/s and max rotational velocity of 50 degrees/s')
    if not len(self._max_cart_vel) == 9:
      rospy.logerr('Length of max_cart_vel does not equal number of joints!')

    self._cart_vel = np.zeros(9)

    # Published velocity message
    self.send_msg = CartVelCmd()
    _dim = [MultiArrayDimension()]
    _dim[0].label = 'cartesian_velocity'
    _dim[0].size = 9
    _dim[0].stride = 9
    self.send_msg.velocity.layout.dim = _dim
    self.send_msg.velocity.data = np.zeros_like(self._cart_vel)
    self.send_msg.header.stamp = rospy.Time.now()
    self.send_msg.header.frame_id = 'joy'

    # Publishers
    self.mode_switch_pub = rospy.Publisher('mode_switches', ModeSwitch, queue_size=1)
    self.modepub = rospy.Publisher("/mi/current_mode", Int16, queue_size=1)

    # Services
    rospy.Service('/teleop_node/set_mode', SetMode, self.set_mode)

    # FUNCTIONS:
  def zero_vel(self):
    for i in range(0,9):
        self._cart_vel[i] = 0

  def set_mode(self, setmode):
    self._mode = setmode.mode_index
    print "Current mode is ", self._mode
    self.publish_mode()
    status = SetModeResponse()
    status = True
    return status

  def publish_mode(self):
    self.modepub.publish(self._mode_key[str(self._mode)])
    print "Current Mode", self._mode_key[str(self._mode)]

  def publish_modeswitch(self):
    self.modeSwitchCount = self.modeSwitchCount+1
    self.modeswitch_msg.header.stamp = rospy.Time.now()
    self.modeswitch_msg.mode = self._mode_key[str(self._mode)]
    self.modeswitch_msg.num_switches = self.modeSwitchCount
    self.mode_switch_pub.publish(self.modeswitch_msg)
    print "Num of mode switches %d" % self.modeSwitchCount

  def change_mode(self, buttons):
    pub_mode = False 
    if self._mode_switching_map == 1:
      if buttons[0]:
          self._mode = (self._mode+1)%(len(self._mode_key))
          pub_mode = True
      elif buttons[1]:
          self._mode = (self._mode-1)%(len(self._mode_key))
          pub_mode = True             
      elif buttons[2]:
          self._mode = len(self._mode_key)-1
          pub_mode = True 
    if self._mode_switching_map == 2 and self._paradigm == 3: 
      if buttons[0]:
          self._mode = 0
          pub_mode = True 
      elif buttons[1]:
          self._mode = 1
          pub_mode = True 
      elif buttons[2]:
          self._mode = 2
          pub_mode = True 
    if pub_mode == True:
      self.publish_modeswitch()
      self.publish_mode()

  def receive(self, msg):
    # handle internal modes
    if not self.waiting_for_release:
        if any(msg.buttons):          
          self.change_mode(msg.buttons)
        self.waiting_for_release = True        
    else:
      if not msg.buttons[0] and not msg.buttons[1]:
        self.waiting_for_release = False
    # zero out the _cart_vel

    self.zero_vel()
    self.handle_velocities(msg)

  def handle_velocities(self, msg):
    # set axes variable
    _axes = np.array(msg.axes)
    # Separate individual axes
    _modes = [int(x) for x in str(self._mode_key[str(self._mode)])]
    # based on mode, find what you want to put _cart_vel
    for i in range(len(_modes)):
        if _modes[i] == 1: # X axis
            self._cart_vel[0] = _axes[i]*self._max_cart_vel[0] * 0.5
        elif _modes[i] == 2: # Y axis, forward y direction is -y, that +1 from joystick should move the robot in -y direction
            self._cart_vel[1] = -_axes[i]*self._max_cart_vel[1] * 0.5
        elif _modes[i] == 3: # Z axis,
            self._cart_vel[2] = _axes[i]*self._max_cart_vel[2] * 0.5
        elif _modes[i] == 4: # Yaw
            self._cart_vel[4] = _axes[i]*self._max_cart_vel[4] * 0.4
        elif _modes[i] == 5: # Pitch
            self._cart_vel[3] = _axes[i]*self._max_cart_vel[3] * 0.4
        elif _modes[i] == 6: # Roll
            self._cart_vel[5] = -_axes[i]*self._max_cart_vel[5] * 0.4
        elif _modes[i] == 7: # gripper mode
          self._cart_vel[6] = _axes[0]*self._max_cart_vel[6]
          self._cart_vel[7] = _axes[0]*self._max_cart_vel[7]
          self._cart_vel[8] = _axes[0]*self._max_cart_vel[8]
    self.send_msg.velocity.data = self._cart_vel
    self.send_msg.header.stamp = rospy.Time.now()
    self.lock.acquire()
    try:
      self.data = self.send_msg
    finally:
      self.lock.release()

  def getDefaultData(self):
    # since this sends velocity, the default will be all zeros.
    self.lock.acquire()
    try:
      self.data = CartVelCmd() # np.zeros_like(self._cart_vel)
    finally:
      self.lock.release()

  def get_mode_switch_paradigm(self):
      switcher = {
        1: {"0":1, "1":2, "2":3, "3":4, "4":5, "5":6, "6":7},
        2: {"0":12, "1":13, "2":45, "3":6, "4":7},
        3: {"0":123, "1":456, "2":7},
      }
      return switcher.get(self._paradigm, "nothing")

  def reconfigure_cb(self, config, level):
    self._paradigm = config.joystick_paradigm
    self._mode_switching_map = config.joystick_mode_switch
    self._mode_key = self.get_mode_switch_paradigm()
    return config

if __name__ == '__main__':
  rospy.init_node('joystick_node', anonymous=True)
  joy = JoystickInput()
  joy.startSend('user_vel')
  rospy.Subscriber('joy', Joy, joy.receive)
  mode_paradigm_srv = Server(ModeSwitchParadigmConfig, joy.reconfigure_cb)

  rospy.spin()
