#!/usr/bin/env python
# Code developed by Deepak Gopinath*, Mahdieh Nejati Javaremi* in February 2020. Copyright (c) 2020. Deepak Gopinath, Mahdieh Nejati Javaremi, Argallab. (*) Equal contribution

import rospy
import collections
import numpy as np
import time
import warnings
import rospkg
import os
import sys

sys.path.append(os.path.join(rospkg.RosPack().get_path("simulators"), "scripts"))


from ds_obs_avoidance.ds_containers.gradient_container import GradientContainer
from ds_obs_avoidance.ds_obstacles.polygon import Cuboid
from ds_obs_avoidance.ds_systems.ds_linear import LinearSystem
from ds_obs_avoidance.ds_avoidance.modulation import obs_avoidance_interpolation_moving
from ds_obs_avoidance.ds_avoidance.utils import obs_check_collision_2d

from sim_pfields.msg import CuboidObs

from sim_pfields.srv import CuboidObsList, CuboidObsListRequest, CuboidObsListResponse
from sim_pfields.srv import AttractorPos, AttractorPosRequest, AttractorPosResponse
from sim_pfields.srv import ComputeVelocity, ComputeVelocityRequest, ComputeVelocityResponse


from adaptive_assistance_sim_utils import *


class SimPFieldsMultiple(object):
    def __init__(self):
        rospy.init_node("sim_pfields_multiple")
        self.environment_dict = collections.OrderedDict()
        self.num_obstacles_dict = collections.OrderedDict()
        self.obs_descs_dict = collections.OrderedDict()

        self.initial_ds_system_dict = collections.OrderedDict()
        self.attractor_orientation_dict = collections.OrderedDict()
        self.vel_scale_factor = 5.0
        self.ROT_VEL_MAGNITUDE = 1.5

        rospy.Service("/sim_pfields_multiple/init_obstacles", CuboidObsList, self.populate_environment)
        rospy.Service("/sim_pfields_multiple/update_ds", AttractorPos, self.update_ds)
        rospy.Service("/sim_pfields_multiple/compute_velocity", ComputeVelocity, self.compute_velocity)

    def populate_environment(self, req):
        # print("In POPULATE ENVIRONMENT service")
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
        # print("ENVIRONMENT ", self.environment_dict[pfield_id].list)
        return response

    def update_ds(self, req):
        # print("In UPDATE DS service")
        pfield_id = req.pfield_id
        attractor_position = req.attractor_position
        attractor_orientation = req.attractor_orientation
        # print("ATTRACTOR POSITION for ", pfield_id, attractor_position)
        # print("ATTRACTOR ORIENTATION for ", pfield_id, attractor_orientation)
        self.initial_ds_system_dict[pfield_id] = LinearSystem(attractor_position=np.array(attractor_position))
        self.attractor_orientation_dict[pfield_id] = attractor_orientation
        response = AttractorPosResponse()
        response.success = True

        return response

    def compute_velocity(self, req):

        current_pos = req.current_pos
        current_orientation = req.current_orientation
        pfield_id = req.pfield_id
        dynamical_system = self.initial_ds_system_dict[pfield_id].evaluate
        obs_avoidance_func = obs_avoidance_interpolation_moving  # the obs function to be used
        attractor_orientation = self.attractor_orientation_dict[pfield_id]
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

        for ix in range(N_x):
            for iy in range(N_y):
                if not indOfNoCollision[ix, iy]:
                    continue
                pos = np.array([XX[ix, iy], YY[ix, iy]])
                xd_init[:, ix, iy] = dynamical_system(pos)  # initial DS
                xd_mod[:, ix, iy] = obs_avoidance_func(pos, xd_init[:, ix, iy], obs)
                # xd_mod[:, ix, iy] = xd_init[:, ix, iy]  # DEBUGGING only!!

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

        # compute rotational velocity
        # both angles in [0, 2pi]
        vel[-1] = self._get_shortest_angle_direction(current_orientation, attractor_orientation)
        response.velocity_final = vel

        return response

    def _get_shortest_angle_direction(self, current_angle, target_angle):
        # ASSUMPTION: both current_angle and target_angle are in [0, 2pi]

        if abs(target_angle - current_angle) >= PI:
            if target_angle > current_angle:
                rotational_velocity = -self.ROT_VEL_MAGNITUDE
            elif current_angle > target_angle:
                rotational_velocity = self.ROT_VEL_MAGNITUDE
        elif abs(target_angle - current_angle) < PI:
            if target_angle > current_angle:
                rotational_velocity = self.ROT_VEL_MAGNITUDE
            elif current_angle > target_angle:
                rotational_velocity = -self.ROT_VEL_MAGNITUDE
            elif current_angle - target_angle < 0.01:
                rotational_velocity = 0.0

        return rotational_velocity


if __name__ == "__main__":
    SimPFieldsMultiple()
    rospy.spin()
