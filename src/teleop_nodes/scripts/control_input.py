#!/usr/bin/env python
# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

import numpy as np
import threading
import rospy
from std_msgs.msg import MultiArrayDimension
from teleop_nodes.msg import CartVelCmd


class ControlInput(object):

    """Abstract base class for controller inputs for ROS"""


    def __init__(self, robot_dim=6, finger_dim=3):
        #init filter
        self.robot_dim = robot_dim
        self.finger_dim = finger_dim
        self.filter_list = [[0]*self.robot_dim] * 3     #filter size = 20
        # print self.filter_list
        self.lock = threading.Lock()
        # self.data = self.getDefaultData()
        self.send_msg = CartVelCmd()
        _dim = [MultiArrayDimension()]
        _dim[0].label = 'cartesian_velocity'
        _dim[0].size = self.robot_dim+self.finger_dim
        _dim[0].stride = self.robot_dim+self.finger_dim
        self.send_msg.velocity.layout.dim = _dim
        #self.send_msg.velocity.layout.data_offset = 0
        self.send_msg.velocity.data = np.zeros_like(np.zeros(9))
        self.lock.acquire()
        try:
            self.data = self.send_msg
        finally:
            self.lock.release()


        # self.finger_maxTurn = 6000
        # self.currentFingerPosition = [0.0, 0.0, 0.0]
        # self.positions = [0, 0, 0]
        # topic_address = '/j2s7s300_driver/out/finger_position'
        # rospy.Subscriber(topic_address, FingerPosition, self.setCurrentFingerPosition)
        # rospy.wait_for_message(topic_address, FingerPosition)
        # print 'obtained current finger position'
        # print self.currentFingerPosition


    # def setCurrentFingerPosition(self, feedback):
    #     # self.lock.acquire()
    #     self.currentFingerPosition[0] = feedback.finger1
    #     self.currentFingerPosition[1] = feedback.finger2
    #     self.currentFingerPosition[2] = feedback.finger3
    #     finger_value = self.data.data[6:]
    #     finger_turn = [finger_value[i] + self.currentFingerPosition[i] for i in range(0, 3)]
    #     positions_temp1 = [max(0.0, n) for n in finger_turn]
    #     positions_temp2 = [min(n, self.finger_maxTurn) for n in positions_temp1]
    #     self.positions = [float(n) for n in positions_temp2]
    #     # print "FINGER VALUES, currentFingerPosition, POSITions ", finger_value, self.currentFingerPosition, self.positions
    #     # self.lock.release()

    def startSend(self, rostopic, period=rospy.Duration(0.01)):
        """
        Start the send thread.

        :param period: Period between sent packets, ``rospy.Duration``
        :param rostopic: name of rostopic to update, ``str``
        """

        self.pub = rospy.Publisher(rostopic, CartVelCmd, queue_size = 1)
        # self.pub = rospy.Publisher(rostopic, PoseVelocity, queue_size=1)
        self.send_thread = threading.Thread(target=self._send, args=(period,))
        self.send_thread.start()


    def _send(self, period):
        while not rospy.is_shutdown():
            start = rospy.get_rostime()

            self.lock.acquire()
            try:
                self.filter_list.pop(0)
                self.filter_list.append(list(self.data.velocity.data[:self.robot_dim]))
                data = CartVelCmd()
                data.velocity.data[:self.robot_dim] = list(np.mean(self.filter_list, axis = 0))
                if self.finger_dim > 0:
                    data.velocity.data[self.robot_dim:] = self.data.velocity.data[self.robot_dim:]
                data.header = self.data.header
            finally:
                self.lock.release()


            self.pub.publish(data)

            end = rospy.get_rostime()

            if end - start < period:
                rospy.sleep(period - (end - start))
            else:
                rospy.logwarn("Sending data took longer than the specified period.")

    def receive(*args, **kwargs):
        """
        Receive method, to be passed to the subscriber as a callback.

        This method is not implemented in the base class because it depends on
        the input type.
        """

        raise NotImplementedError

    def getDefaultData(*args, **kwargs):
        """
        Returns whatever message should be sent when no data has been received.

        (must be a CartVelCmd)
        """
        raise NotImplementedError
