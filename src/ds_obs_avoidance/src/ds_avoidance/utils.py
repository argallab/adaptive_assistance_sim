"""
Varios tools and uitls for advanced obstacle-avoidance-library
"""
# Author Lukas Huber
# Date 2018-02-15

import warnings

import numpy as np
import numpy.linalg as LA
from numpy import pi

from ds_utils.angle_math import *
from ds_utils.linalg import get_orthogonal_basis

from ds_utils.directional_space import get_directional_weighted_sum
