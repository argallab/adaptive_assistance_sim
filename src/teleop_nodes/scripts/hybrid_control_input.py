#!/usr/bin/env python
# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

import numpy as np
import threading
import rospy
from teleop_nodes.msg import InterfaceSignal


class HybridControlInput(object):
    def __init__(self, interface_velocity_dim=1):
        self.interface_velocity_dim = interface_velocity_dim
        self.filter_length = 3
        # filter just to smooth the velocity action. Mode switches are not filtered.
        self.filter_list = [[0] * self.interface_velocity_dim] * self.filter_length
        self.lock = threading.Lock()

        self.send_msg = InterfaceSignal()
        self.send_msg.interface_signal = [0] * self.interface_velocity_dim

        self.lock.acquire()
        try:
            self.data = self.send_msg
        finally:
            self.lock.release()

    def startSend(self, rostopic, period=rospy.Duration(0.005)):
        """
        Start the send thread.

        :param period: Period between sent packets, ``rospy.Duration``
        :param rostopic: name of rostopic to update, ``str``
        """

        self.pub = rospy.Publisher(rostopic, InterfaceSignal, queue_size=1)
        # self.pub = rospy.Publisher(rostopic, PoseVelocity, queue_size=1)
        self.send_thread = threading.Thread(target=self._send, args=(period,))
        self.send_thread.start()

    def _send(self, period):
        while not rospy.is_shutdown():
            start = rospy.get_rostime()

            self.lock.acquire()
            try:
                self.filter_list.pop(0)
                self.filter_list.append(list(self.data.interface_signal[: self.interface_velocity_dim]))
                data = InterfaceSignal()
                data.interface_signal[: self.interface_velocity_dim] = list(np.mean(self.filter_list, axis=0))
                data.mode_switch = self.data.mode_switch
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

        (must be a InterfaceSignal)
        """
        raise NotImplementedError
