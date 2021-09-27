#!/usr/bin/env python
# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

import rospy
import numpy as np
import time
import warnings

from ds_obs_avoidance.ds_containers.gradient_container import GradientContainer
from ds_obs_avoidance.ds_obstacles.polygon import Cuboid
from ds_obs_avoidance.ds_systems.ds_linear import LinearSystem
from ds_obs_avoidance.ds_avoidance.modulation import obs_avoidance_interpolation_moving
from ds_obs_avoidance.ds_avoidance.utils import obs_check_collision_2d

from sim_pfields.msg import CuboidObs

from sim_pfields.srv import CuboidObsList, CuboidObsListRequest, CuboidObsListResponse
from sim_pfields.srv import AttractorPos, AttractorPosRequest, AttractorPosResponse
from sim_pfields.srv import ComputeVelocity, ComputeVelocityRequest, ComputeVelocityResponse


class SimPFields(object):
    def __init__(self):
        rospy.init_node("sim_pfields")
        self.environment = GradientContainer()
        self.num_obstacles = 0
        self.obs_descs = []

        self.initial_ds_system = None
        self.vel_scale_factor = 2.0

        rospy.Service("/sim_pfields/init_obstacles", CuboidObsList, self.populate_environment)
        rospy.Service("/sim_pfields/update_ds", AttractorPos, self.update_ds)
        rospy.Service("/sim_pfields/compute_velocity", ComputeVelocity, self.compute_velocity)

    def populate_environment(self, req):
        print("In POPULATE ENVIRONMENT service")
        self.num_obstacles = req.num_obstacles
        self.obs_descs = req.obs_descs  # List of CuboidObss
        assert self.num_obstacles == len(self.obs_descs)
        self.environment = GradientContainer()
        for i in range(self.num_obstacles):
            # if no obstacles then self.environment will be []
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
        print("ENVIRONMENT ", self.environment.list)
        return response

    def update_ds(self, req):
        print("In UPDATE DS service")
        attractor_position = req.attractor_position
        attractor_orientation = req.attractor_orientation
        print("ATTRACTOR POSITION ", attractor_position)

        # set up the translational dynamical system
        self.initial_ds_system = None
        self.initial_ds_system = LinearSystem(attractor_position=np.array(attractor_position))
        response = AttractorPosResponse()
        response.success = True
        return response

    def compute_velocity(self, req):

        current_pos = req.current_pos
        dynamical_system = self.initial_ds_system.evaluate
        obs_avoidance_func = obs_avoidance_interpolation_moving
        pos_attractor = self.initial_ds_system.attractor_position
        obs = self.environment

        dim = 2
        vector_field_only_outside = True

        N_x = N_y = 1
        XX, YY = np.array([[current_pos[0]]]), np.array([[current_pos[1]]])

        if vector_field_only_outside:
            if hasattr(obs, "check_collision_array"):
                pos = np.vstack((XX.flatten(), YY.flatten()))
                collision_index = obs.check_collision_array(pos)
                indOfNoCollision = np.logical_not(collision_index).reshape(N_x, N_y)
            else:
                warnings.warn("Depreciated (non-attribute) collision method.")
                indOfNoCollision = obs_check_collision_2d(obs, XX, YY)
        else:
            indOfNoCollision = np.ones((N_x, N_y))

        xd_init = np.zeros((2, N_x, N_y))
        xd_mod = np.zeros((2, N_x, N_y))
        t_start = time.time()

        t_start = time.time()

        for ix in range(N_x):
            for iy in range(N_y):
                if not indOfNoCollision[ix, iy]:
                    continue
                pos = np.array([XX[ix, iy], YY[ix, iy]])
                xd_init[:, ix, iy] = dynamical_system(pos)  # initial DS
                xd_mod[:, ix, iy] = obs_avoidance_func(pos, xd_init[:, ix, iy], obs)
                # xd_mod[:, ix, iy] = xd_init[:, ix, iy]  # DEBUGGING only!!
        t_end = time.time()
        n_collfree = np.sum(indOfNoCollision)

        if not n_collfree:  # zero points
            warnings.warn("No collision free points in space.")
        else:
            pass
            # print("Average time per evaluation {} ms".format(round((t_end - t_start) * 1000 / (n_collfree), 3)))

        dx1_noColl, dx2_noColl = np.squeeze(xd_mod[0, :, :]), np.squeeze(xd_mod[1, :, :])
        # end_time = time.time()
        # n_calculations = np.sum(indOfNoCollision)
        # print("Number of free points: {}".format(n_calculations))
        # print("Average time: {} ms".format(np.round((end_time - start_time) / (n_calculations) * 1000), 5))
        # print("Modulation calculation total: {} s".format(np.round(end_time - start_time), 4))

        response = ComputeVelocityResponse()
        vel = np.array([float(dx1_noColl), float(dx2_noColl), 0.0])
        vel = vel / np.linalg.norm(vel)
        response.velocity_final = vel

        return response


if __name__ == "__main__":
    SimPFields()
    rospy.spin()
