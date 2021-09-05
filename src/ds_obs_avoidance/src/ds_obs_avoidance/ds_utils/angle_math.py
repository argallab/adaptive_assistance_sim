"""
Angle math for python in 2D
Helper function for directional & angle evaluations
"""

import warnings
from math import pi

import numpy as np


def angle_is_between(angle_test, angle_low, angle_high):
    """ Verify if angle_test is in between angle_low & angle_high """
    delta_low = angle_difference_directional(angle_test, angle_low)
    delta_high = angle_difference_directional(angle_high, angle_test)

    return delta_low > 0 and delta_high > 0


def angle_is_in_between(angle_test, angle_low, angle_high, margin=1e-9):
    """ Verify if angle_test is in between angle_low & angle_high
    Values are between [0, 2pi]. 
    Margin to account for numerical errors. """
    delta_low = angle_difference_directional_2pi(angle_test, angle_low)
    delta_high = angle_difference_directional_2pi(angle_high, angle_test)

    delta_tot = angle_difference_directional_2pi(angle_high, angle_low)

    return np.abs((delta_high + delta_low) - delta_tot) < margin


def angle_modulo(angle):
    """ Get angle in [-pi, pi[  """
    return ((angle + pi) % (2 * pi)) - pi


def angle_difference_directional_2pi(angle1, angle2):
    angle_diff = angle1 - angle2
    while angle_diff > 2 * pi:
        angle_diff -= 2 * pi
    while angle_diff < 0:
        angle_diff += 2 * pi
    return angle_diff


def angle_difference_directional(angle1, angle2):
    """
    Difference between two angles ]-pi, pi]
    Note: angle1-angle2 (non-commutative)
    """
    angle_diff = angle1 - angle2
    while angle_diff > pi:
        angle_diff = angle_diff - 2 * pi
    while angle_diff <= -pi:
        angle_diff = angle_diff + 2 * pi
    return angle_diff


def angle_difference(angle1, angle2):
    return angle_difference_directional(angle1, angle2)


def angle_difference_abs(angle1, angle2):
    """
    Difference between two angles [0,pi[
    angle1-angle2 = angle2-angle1(commutative)
    """
    angle_diff = np.abs(angle2 - angle1)
    while angle_diff >= pi:
        angle_diff = 2 * pi - angle_diff
    return angle_diff


def transform_polar2cartesian(magnitude, angle, center_position=None):
    """ Transform 2d from polar- to cartesian coordinates."""
    """
    center_position is the origin for the coordinate system in which (r, theta) is defined

    """

    # Only 2D input
    if center_position is None:
        # currently only for 2D
        center_position = np.zeros(2)
    else:
        if isinstance(center_position, np.ndarray):
            pass
        elif isinstance(center_position, list):
            center_position = np.array(center_position)
        else:
            raise TypeError("Wrong input type vector")

        assert center_position.shape[0] == 2

    # (r, theta) for polar coordinate
    magnitude = np.reshape(magnitude, (-1))  # (1, )
    angle = np.reshape(angle, (-1))  # (1, )

    if center_position is None:
        cartesian_coords = magnitude * np.vstack((np.cos(angle), np.sin(angle)))
    else:
        # (2, 1)
        cartesian_coords = (
            magnitude * np.vstack((np.cos(angle), np.sin(angle))) + np.tile(center_position, (magnitude.shape[0], 1)).T
        )

    return np.squeeze(cartesian_coords)  # (2, )


def transform_cartesian2polar(cartesian_coords, center_position=None, second_axis_is_dim=True):
    """
    Two dimensional transformation of cartesian to polar coordinates
    Based on center_position (default value center_position=np.zeros(dim))
    center_position = location of origin of the target polar coordinate system
    """
    # TODO -- check dim and etc
    # Don't just squeeze, maybe...

    # if type(center_position)==type(None):
    # center_position = np.zeros(self.dim)

    cartesian_coords = np.squeeze(cartesian_coords)  # (dim, )
    if second_axis_is_dim:
        # if cartesian_coords in [1, dim]
        cartesian_coords = cartesian_coords.T  # (dim, 1)
    dim = cartesian_coords.shape[0]

    if isinstance(center_position, type(None)):
        center_position = np.zeros(dim)
    else:
        center_position = np.squeeze(center_position)  # (dim, 1)
        if second_axis_is_dim:
            center_position = center_position.T
        assert center_position.shape[0] == dim

    if len(cartesian_coords.shape) == 1:  # (dim, )
        cartesian_coords = cartesian_coords - center_position  # subtract the position from target origin
        angle = np.arctan2(cartesian_coords[1], cartesian_coords[0])  # compute angle
    else:
        cartesian_coords = cartesian_coords - np.tile(center_position, (cartesian_coords.shape[1], 1)).T
        angle = np.arctan2(cartesian_coords[1, :], cartesian_coords[0, :])

    magnitude = np.linalg.norm(cartesian_coords, axis=0)  # find length to compute r

    # output: [r, phi]
    return magnitude, angle

def periodic_weighted_sum(angles, weights, reference_angle=None):
    """Weighted Average of angles (1D)"""
    # TODO: unify with directional_weighted_sum() // see above
    # Extend to dimenions d>2
    if isinstance(angles, list): 
        angles = np.array(angles)
    if isinstance(weights, list): 
        weights = np.array(weights)

    
    if reference_angle is None:
        if len(angles)>2:
            raise NotImplementedError("No mean defined for periodic function with more than two angles.")
        reference_angle = angle_difference_directional(angles[0], angles[1])/2.0 + angles[1]
        reference_angle = angle_modulo(reference_angle)

    angles = angle_modulo(angles-reference_angle)
    
    mean_angle = angles.T.dot(weights)
    mean_angle = angle_modulo(mean_angle + reference_angle)

    return mean_angle
