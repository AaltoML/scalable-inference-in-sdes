"""Construct the weights sigma points of various sigma point approximations."""

import tensorflow as tf
import numpy as np


def get_unscented_sigma_points(dim=6, lamda=1):
    """Get unscented sigma points.

    Inputs
    -----
        dim: int,
            Number of dimensions
        lamba: numeric,
            Scales the unit vectors
    Outputs
    ------
        tf.Tensor: A tensor consisting of all
            the 2*dim sigma points

    """
    unit_vectors = tf.convert_to_tensor([[0]*i + [1] + [0]*(dim - 1 - i) for i in range(dim)], tf.float64)
    mult_term = np.sqrt(lamda + dim)
    zeros = tf.zeros((1, dim), dtype=tf.float64)
    sigmas = tf.sqrt(mult_term)*unit_vectors
    sigmas_neg = -tf.sqrt(mult_term)*unit_vectors

    return tf.concat([zeros, sigmas, sigmas_neg], axis=0)


def get_unscented_weights(dim=6, kappa=1, lamda=1):
    """Get weights for the unscented sigma points.

    Inputs
    ------
        dim: int,
            Number of dimensions
        kappa: float,
            A scaling constant
        lamda: float,
            A scaling constant, same as used
            for the sigma points.
    Outputs
    ------
        tf.Tensor: A tensor of length 2*dim,
            with weights for each sigma point.
    """
    w0 = lamda/(dim+kappa)
    w_i = 1/(2*(dim+kappa))
    w = tf.convert_to_tensor([w0] + [w_i]*(2*dim), dtype=tf.float64)
    return w


def get_cubature_sigma_points(dim):
    """Get the cubature sigma points  for a given dimension.

    Inputs
    ------
        dim: int,
            Number of dimensions.
    Outputs
    ------
        tf.Tensor: A tensor consisting of
        the 2*dim sigma points.
    """
    unit_vectors = tf.convert_to_tensor([[0]*i + [1] + [0]*(dim - 1 - i) for i in range(dim)], tf.float64)
    dim_tf = tf.convert_to_tensor(dim, tf.float64)
    sigmas = tf.sqrt(dim_tf)*unit_vectors
    sigmas_neg = -tf.sqrt(dim_tf)*unit_vectors
    return tf.concat([sigmas, sigmas_neg], axis=0)


def get_cubature_weights(dim):
    """Get weights for the cubature sigma points.

    Inputs
    -----
        dim: int,
            Number of dimensions
    Outputs
    -------
        np.array: An array of length 2*dim
            consisting of the sigma point
            weights.
    """
    w = np.ones(2*dim)/(2*dim)
    return w
