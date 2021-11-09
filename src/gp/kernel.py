"""
Curl-free and divergence-free GPFlow compatible kernel implementation.
"""

import tensorflow as tf

import gpflow
from gpflow.base import Parameter
from gpflow.utilities import positive
from gpflow.utilities.ops import square_distance, difference_matrix


class CurlFreeKernel(gpflow.kernels.MultioutputKernel):
    def __init__(self, output_dim, variance=1.0, lengthscale=1.0):
        super(CurlFreeKernel, self).__init__()
        self.variance = Parameter(variance, transform=positive())
        self.lengthscale = Parameter(lengthscale, transform=positive())
        self.output_dim = output_dim

    @property
    def num_latent_gps(self):
        pass

    @property
    def latent_kernels(self):
        pass

    def scale(self, X):
        return X / self.lengthscale if X is not None else X

    def K(self, X, X2=None, full_output_cov=True):
        if X2 is None:
            X2 = X

        dist = square_distance(self.scale(X), self.scale(X2))
        K2 = tf.exp(0.5 * -dist)
        K2 = tf.expand_dims(tf.expand_dims(K2, -1), -1)

        diff = difference_matrix(self.scale(X), self.scale(X2))

        diff1 = tf.expand_dims(diff, -1)
        diff2 = tf.transpose(tf.expand_dims(diff, -1), perm=[0, 1, 3, 2])
        K1_term = diff1 @ diff2

        I = tf.eye(
            self.output_dim, batch_shape=[X.shape[0], X2.shape[0]], dtype=tf.float64
        )
        K1 = (self.variance / tf.square(self.lengthscale)) * (I - K1_term)

        K = K1 * K2

        if full_output_cov:
            return tf.transpose(K, [0, 2, 1, 3])
        else:
            K = tf.linalg.diag_part(K)
            return tf.transpose(K, [2, 0, 1])

    def K_diag(self, X, full_output_cov=True):
        if full_output_cov:
            return self.K(X, full_output_cov=full_output_cov)
        else:
            k = self.K(X, full_output_cov=full_output_cov)
            return tf.transpose(tf.linalg.diag_part(k))


class DivergenceFreeKernel(gpflow.kernels.MultioutputKernel):
    def __init__(self, output_dim, variance=1.0, lengthscale=1.0):
        super(DivergenceFreeKernel, self).__init__()
        self.variance = Parameter(variance, transform=positive())
        self.lengthscale = Parameter(lengthscale, transform=positive())
        self.output_dim = output_dim

    @property
    def num_latent_gps(self):
        pass

    @property
    def latent_kernels(self):
        pass

    def scale(self, X):
        return X / self.lengthscale if X is not None else X

    def K(self, X, X2=None, full_output_cov=True):
        if X2 is None:
            X2 = X

        D = X.shape[1]

        dist = square_distance(self.scale(X), self.scale(X2))
        K2 = tf.exp(0.5 * -dist)
        K2 = tf.expand_dims(tf.expand_dims(K2, -1), -1)

        diff = difference_matrix(self.scale(X), self.scale(X2))

        diff1 = tf.expand_dims(diff, -1)
        diff2 = tf.transpose(tf.expand_dims(diff, -1), perm=[0, 1, 3, 2])
        K1_term = diff1 @ diff2

        K3 = tf.cast(D, dtype=tf.float64) - 1.0 - dist
        K3 = tf.expand_dims(tf.expand_dims(K3, -1), -1)
        K3 = K3 * tf.eye(D, batch_shape=[X.shape[0], X2.shape[0]], dtype=tf.float64)

        K = (K1_term + K3) * K2
        K = (self.variance / tf.square(self.lengthscale)) * K

        if full_output_cov:
            return tf.transpose(K, [0, 2, 1, 3])
        else:
            K = tf.linalg.diag_part(K)
            return tf.transpose(K, [2, 0, 1])

    def K_diag(self, X, full_output_cov=True):
        if full_output_cov:
            return self.K(X, full_output_cov=full_output_cov)
        else:
            k = self.K(X, full_output_cov=full_output_cov)
            return tf.transpose(tf.linalg.diag_part(k))
