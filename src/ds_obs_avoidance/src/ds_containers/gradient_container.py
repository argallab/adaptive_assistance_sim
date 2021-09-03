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

from ds_obstacles.containers import ObstacleContainer
from dynamic_obstacle_avoidance.obstacles import CircularObstacle
from dynamic_obstacle_avoidance.avoidance.utils import get_reference_weight
from dynamic_obstacle_avoidance.avoidance.obs_common_section import *
from dynamic_obstacle_avoidance.avoidance.obs_dynamic_center_3d import *
