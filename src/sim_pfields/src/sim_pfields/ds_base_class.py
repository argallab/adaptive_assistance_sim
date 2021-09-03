"""
Dynamical Systems with a closed-form description.
"""
# Author: Lukas Huber (original), Deepak Gopinath (adaptation to Py2.7)
# Email: hubernikus@gmail.com
# License: BSD (c) 2021

import numpy as np


class DynamicalSystem(object):
    def __init__(self, center_position=None, maximum_velocity=None, dimension=None, attractor_position=None):
        if center_position is None:
            self.center_position = np.zeros(dimension)
        else:
            self.center_position = np.array(center_position)
            dimension = self.center_position.shape[0]
        self.maximum_velocity = maximum_velocity

        if dimension is None and attractor_position is not None:
            dimension = attractor_position.shape[0]
        self.attractor_position = attractor_position

        if dimension is None:
            raise ValueError("Space dimension cannot be guess from inputs. Please define it at initialization.")

        self.dimension = dimension

    @property
    def attractor_position(self):
        return self._attractor_position

    @attractor_position.setter
    def attractor_position(self, value):
        print("Setting attractor position at", value)
        self._attractor_position = value

    def limit_velocity(self, velocity, maximum_velocity=None):
        if maximum_velocity is None:
            if self.maximum_velocity is None:
                return velocity
            else:
                maximum_velocity = self.maximum_velocity

        mag_vel = np.linalg.norm(velocity)
        if mag_vel > maximum_velocity:
            velocity = velocity / mag_vel * maximum_velocity
        return velocity
