"""
Dynamical Systems with a closed-form description.
"""
# Author: Lukas Huber (original), Deepak Gopinath (adaptation to Py2.7)
# Email: hubernikus@gmail.com
# License: BSD (c) 2021

import abc
import numpy as np


class DynamicalSystem(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, center_position=None, maximum_velocity=None, dimension=None, attractor_position=None):
        if center_position is None:
            self.center_position = np.zeros(dimension)
        else:
            self.center_position = np.array(center_position)
            dimension = self.center_position.shape[0]
        # maximum velocity generated
        self.maximum_velocity = maximum_velocity

        # if attractor position is list convert to numpy
        if type(attractor_position) is list:
            attractor_position = np.array(attractor_position)

        if dimension is None and attractor_position is not None:
            dimension = attractor_position.shape[0]

        self.attractor_position = attractor_position

        if dimension is None:
            raise ValueError("Space dimension cannot be guess from inputs. Please define it at initialization.")

        self.dimension = dimension

    @property
    def attractor_position(self):
        # todo remove print
        print("In getter for attractor position")
        return self._attractor_position

    @attractor_position.setter
    def attractor_position(self, value):
        # todo remove print
        print("Setting attractor position at", value)
        self._attractor_position = value

    def limit_velocity(self, velocity, maximum_velocity=None):
        if maximum_velocity is None:
            if self.maximum_velocity is None:
                # if no maximum velocity is set just return the current velocity
                return velocity
            else:
                maximum_velocity = self.maximum_velocity

        # scale velocity to maximum velocity
        mag_vel = np.linalg.norm(velocity)
        if mag_vel > maximum_velocity:
            velocity = velocity / mag_vel * maximum_velocity
        return velocity

    @abc.abstractmethod
    def evaluate(self, position):
        """ Returns velocity of the evaluated the dynamical system at 'position'."""
        pass

    def compute_dynamics(self, position):
        # This  or 'evaluate' / to be or not to be?!
        pass

    def check_convergence(self, *args, **kwargs):
        """ Non compulsary function (only for stable systems), but needed to stop integration. """
        raise NotImplementedError("No convergence check implemented.")

    def evaluate_array(self, position_array):
        """ Return an array of positions evluated. """
        velocity_array = np.zeros(position_array.shape)
        for ii in range(position_array.shape[1]):
            velocity_array[:, ii] = self.evaluate_array(position_array[:, ii])
        return velocity_array

    def motion_integration(self, start_position, dt, max_iteration=10000):
        """ Integrate spiral Motion """
        dataset = []
        dataset.append(start_position)
        current_position = start_position

        it_count = 0
        while True:
            it_count += 1
            if it_count > max_iteration:
                print("Maximum number of iterations reached")
                break

            if self.check_convergence(dataset[-1]):
                print("Trajectory converged to goal..")
                break

            # get velocity at current position
            delta_vel = self.evaluate(dataset[-1])
            # euler integration for computing new position
            current_position = delta_vel * dt + current_position
            # update trajectory list
            dataset.append(current_position)

        return np.array(dataset).T  # convert to np.array and transpose
