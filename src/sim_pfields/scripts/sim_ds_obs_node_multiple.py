#!/usr/bin/env python
# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

import rospy
import collections
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


class SimPFieldsMultiple(object):
    def __init__(self):
        rospy.init_node("sim_pfields_multiple")
        self.environment_dict = collections.OrderedDict()
        self.num_obstacles_dict = collections.OrderedDict()
        self.obs_descs_dict = collections.OrderedDict()

        self.initial_ds_system_dict = collections.OrderedDict()
        self.vel_scale_factor = 5.0

        rospy.Service("/sim_pfields_multiple/init_obstacles", CuboidObsList, self.populate_environment)
        rospy.Service("/sim_pfields_multiple/update_ds", AttractorPos, self.update_ds)
        rospy.Service("/sim_pfields_multiple/compute_velocity", ComputeVelocity, self.compute_velocity)

    def populate_environment(self, req):
        print("In POPULATE ENVIRONMENT service")
        pfield_id = req.pfield_id
        num_obstacles = req.num_obstacles
        self.obs_descs_dict[pfield_id] = req.obs_descs  # List of CuboidObss
        assert num_obstacles == len(self.obs_descs_dict[pfield_id])
        self.environment_dict[pfield_id] = GradientContainer()
        for i in range(num_obstacles):
            # if no obstacles then self.environment will be []
            obs_desc = self.obs_descs_dict[pfield_id][i]  # CuboidObs
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
            self.environment_dict[pfield_id].append(cuboid_obs)

        response = CuboidObsListResponse()
        response.status = True
        print("ENVIRONMENT ", self.environment_dict[pfield_id].list)
        return response

    def update_ds(self, req):
        print("In UPDATE DS service")
        pfield_id = req.pfield_id
        attractor_position = req.attractor_position
        print("ATTRACTOR POSITION ", attractor_position)
        self.initial_ds_system_dict[pfield_id] = LinearSystem(attractor_position=np.array(attractor_position))
        response = AttractorPosResponse()
        response.success = True
        print("Current inital ds dict ", self.initial_ds_system_dict)
        return response

    def compute_velocity(self, req):

        current_pos = req.current_pos
        pfield_id = req.pfield_id
        dynamical_system = self.initial_ds_system_dict[pfield_id].evaluate
        obs_avoidance_func = obs_avoidance_interpolation_moving
        pos_attractor = self.initial_ds_system_dict[pfield_id].attractor_position
        obs = self.environment_dict[pfield_id]

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
        vel = self.vel_scale_factor * (vel / np.linalg.norm(vel))
        response.velocity_final = vel

        return response


if __name__ == "__main__":
    SimPFieldsMultiple()
    rospy.spin()
