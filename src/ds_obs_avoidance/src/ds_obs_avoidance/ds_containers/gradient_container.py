"""
Container encapsulates all obstacles.
Gradient container finds the dynamic reference point through gradient descent.
"""
# Author: "LukasHuber"
# Email: lukas.huber@epfl.ch
# License: BSD (c) 2021

import warnings, sys
import numpy as np
import copy
import time

from ds_obs_avoidance.ds_containers.container import ObstacleContainer
from ds_obs_avoidance.ds_obstacles.obstacles import CircularObstacle
from ds_obs_avoidance.ds_avoidance.utils import get_reference_weight
from ds_obs_avoidance.ds_avoidance.obs_dynamic_center_3d import *
from ds_obs_avoidance.ds_avoidance.obs_common_section import *


class GradientContainer(ObstacleContainer):
    def __init__(self, obs_list=None):
        super(GradientContainer, self).__init__(obs_list)

        self._obstacle_is_updated = np.ones(self.number, dtype=bool)

        if len(self):
            self._boundary_reference_points = np.zeros((self.dim, len(self), len(self)))
            self._distance_matrix = DistanceMatrix(n_obs=len(self))
