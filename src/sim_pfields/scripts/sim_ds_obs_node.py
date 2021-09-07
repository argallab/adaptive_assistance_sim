#!/usr/bin/env python
# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

import rospy
import os
import numpy as np
import math
import sys
import collections


from ds_obs_avoidance.ds_containers.gradient_container import GradientContainer
from ds_obs_avoidance.ds_obstacles.polygon import Cuboid
from ds_obs_avoidance.ds_systems.ds_linear import LinearSystem

from sim_pfields.msg import CuboidObs

from sim_pfields.srv import CuboidObsList, CuboidObsListRequest, CuboidObsListResponse
from sim_pfields.srv import AttractorPos, AttractorPosRequest, AttractorPosResponse


class SimPFields(object):
    def __init__(self):
        rospy.init_node("sim_pfields")
        self.environment = GradientContainer()
        self.num_obstacles = 0
        self.obs_descs = []

        self.initial_ds_system = None

        rospy.Service("/sim_pfields/init_obstacles", CuboidObsList, self.populate_environment)
        rospy.Service("sim_pfields/update_ds", AttractorPos, self.update_ds)

    def populate_environment(self, req):
        self.num_obstacles = req.num_obstacles
        self.obs_descs = req.obs_descs  # List of CuboidObss
        assert self.num_obstacles == len(self.obs_descs)
        for i in range(self.num_obstacles):
            obs_desc = self.obs_descs[i]  # CuboidObs
            center_position = obs_desc.center_position
            orientation = obs_desc.orientation
            axes_length = obs_desc.axes_length
            is_boundary = obs_desc.is_boundary

            cuboid_obs = Cuboid(
                center_position=center_position,
                orientation=orientation,
                axes_length=axes_length,
                is_boundary=is_boundary,
            )
            self.environment.append(cuboid_obs)

        response = CuboidObsListResponse()
        response.status = True

        return response

    def update_ds(self, req):
        attractor_position = req.attractor_position
        self.initial_ds_system = None
        self.initial_ds_system = LinearSystem(attractor_position=np.array(attractor_position))
        response = AttractorPosResponse()
        response.status = True
        return True


if __name__ == "__main__":
    SimPFields()
    rospy.spin()
