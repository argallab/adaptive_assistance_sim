"""
Different linear algebraig helper function (mainly) based on numpy
"""

import numpy as np


def is_positive_definite(x):
    """ Check if input matrix x is positive definite and return True/False."""
    return np.all(np.linalg.eigvals(x) > 0)


def is_negative_definite(x):
    """ Check if input matrix x is positive definite and return True/False."""
    return np.all(np.linalg.eigvals(x) < 0)


def get_orthogonal_basis(vector, normalize=True):
    """ Get Orthonormal basis matrix for an dimensional input vector. """

    # convert to np.array
    if isinstance(vector, np.ndarray):
        pass
    elif isinstance(vector, list):
        vector = np.array(vector)
    else:
        raise TypeError("Wrong input type vector")

    if normalize:
        # normalize input vector to have unit norm
        v_norm = np.linalg.norm(vector)
        if v_norm:
            vector = vector / v_norm
        else:
            raise ValueError("Orthogonal basis Matrix not defined for 0-direction vector.")

    dim = vector.shape[0]  # dimensionality of the vector
    basis_matrix = np.zeros((dim, dim))  # basis matrix is a square matrix of size dim by dim

    if dim == 2:
        basis_matrix[:, 0] = vector
        basis_matrix[:, 1] = np.array([-basis_matrix[1, 0], basis_matrix[0, 0]])
    elif dim == 3:
        basis_matrix[:, 0] = vector
        basis_matrix[:, 1] = np.array([-vector[1], vector[0], 0])

        norm_vec2 = np.linalg.norm(basis_matrix[:, 1])
        if norm_vec2:
            basis_matrix[:, 1] = basis_matrix[:, 1] / norm_vec2
        else:
            basis_matrix[:, 1] = [1, 0, 0]

        basis_matrix[:, 2] = np.cross(basis_matrix[:, 0], basis_matrix[:, 1])

        norm_vec = np.linalg.norm(basis_matrix[:, 2])
        if norm_vec:
            basis_matrix[:, 2] = basis_matrix[:, 2] / norm_vec
    elif dim > 3:
        basis_matrix[:, 0] = vector
        for ii in range(1, dim):
            # TODO: higher dimensions
            if vector[ii]:  # nonzero
                basis_matrix[:ii, ii] = vector[:ii]
                basis_matrix[ii, ii] = -np.sum(vector[:ii] ** 2) / vector[ii]
                basis_matrix[: ii + 1, ii] = basis_matrix[: ii + 1, ii] / np.linalg.norm(basis_matrix[: ii + 1, ii])
            else:
                basis_matrix[ii, ii] = 1
            # basis_matrix[dim-(ii), ii] = -np.dot(vector[:dim-(ii)], vector[:dim-(ii)])
            # basis_matrix[:, ii] = basis_matrix[:, ii]/LA.norm(basis_matrix[:, ii])

        # raise ValueError("Not implemented for d>3")
        # warnings.warn("Implement higher dimensionality than d={}".format(dim))

    return basis_matrix
