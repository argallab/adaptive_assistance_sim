""" Container to describe obstacles & wall environemnt"""
# Author Lukas Huber
# Mail lukas.huber@epfl.ch
# Created 2021-06-22
# License: BSD (c) 2021

import numpy as np
import warnings

from ds_obs_avoidance.ds_avoidance.utils import *


class BaseContainer(list):
    def __init__(self, obs_list=None):
        self._obstacle_list = []

        if isinstance(obs_list, (list, BaseContainer)):
            self._obstacle_list = obs_list

    def __getitem__(self, key):
        """ List-like or dictionarry-like access to obstacle"""
        # TODO: can this be done more efficiently?
        if isinstance(key, (str)):
            for ii in range(len(self._obstacle_list)):
                if self._obstacle_list[ii].name == key:
                    return self._obstacle_list[ii]
            raise ValueError("Obstacle <<{}>> not in list.".format(key))
        else:
            return self._obstacle_list[key]

    def __setitem__(self, key, value):
        self._obstacle_list[key] = value

    def append(self, value):  # Compatibility with normal list.
        """ Add new elements to obstacles list. The wall obstacle is placed last."""
        self._obstacle_list.append(value)

    def __delitem__(self, key):
        """Obstacle is not part of the workspace anymore."""
        del self._obstacle_list[key]

    def add_obstacle(self, value):
        self.append(value)

    def __iter__(self):
        return iter(self._obstacle_list)

    def __repr__(self):
        return "ObstacleContainer of length #{}".format(len(self))

    def __str__(self):
        return "ObstacleContainer of length #{}".format(len(self))

    def __len__(self):
        return len(self._obstacle_list)

    @property
    def boundary(self):
        return self._obstacle_list[-1]

    @property
    def number(self):
        return len(self)

    @property
    def n_obstacles(self):
        return len(self)

    @property
    def dimension(self):
        # Dimension of all obstacles is expected to be equal
        return self._obstacle_list[0].dim

    @property
    def dim(self):
        # Dimension of all obstacles is expected to be equal
        return self._obstacle_list[0].dim

    @property
    def list(self):
        return self._obstacle_list

    @property
    def has_environment(self):
        return bool(len(self))