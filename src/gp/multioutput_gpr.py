# This is a copy of GPFlow's GPR with a few changes to make it work with multioutput kernel - curlfree and divergencefree.

# Copyright 2016 James Hensman, Valentine Svensson, alexggmatthews, fujiisoup
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple

import tensorflow as tf

import gpflow
from gpflow.kernels import Kernel
from gpflow.models.gpr import multivariate_normal
from gpflow.models.gpr import MeanFunction
from gpflow.models.model import GPModel, InputData, RegressionData, MeanAndVariance
from gpflow.models.gpr import InternalDataTrainingLossMixin


class MultiOutputGPR(GPModel, InternalDataTrainingLossMixin):
    r"""
    Gaussian Process Regression.

    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently.

    The log likelihood of this model is sometimes referred to as the 'log
    marginal likelihood', and is given by

    .. math::
       \log p(\mathbf y \,|\, \mathbf f) =
            \mathcal N(\mathbf{y} \,|\, 0, \mathbf{K} + \sigma_n \mathbf{I})
    """

    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        mean_function: Optional[MeanFunction] = None,
        noise_variance: float = 1.0,
    ):
        likelihood = gpflow.likelihoods.Gaussian(noise_variance)
        _, Y_data = data
        super().__init__(
            kernel, likelihood, mean_function, num_latent_gps=Y_data.shape[-1]
        )
        self.data = data

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.log_marginal_likelihood()

    def log_marginal_likelihood(self) -> tf.Tensor:
        r"""
        Computes the log marginal likelihood.

        .. math::
            \log p(Y | \theta).

        """

        X, Y = self.data

        P = X.shape[1]
        N = X.shape[0]

        K = self.kernel(X)
        K = tf.reshape(K, (P * N, P * N))  # Change 2
        num_data = P * N  # Change 1
        k_diag = tf.linalg.diag_part(K)
        s_diag = tf.fill([num_data], self.likelihood.variance)  # Change

        ks = tf.linalg.set_diag(K, k_diag + s_diag)
        L = tf.linalg.cholesky(ks)

        X = tf.repeat(X, P, axis=0)  # Change 4

        m = self.mean_function(X)

        # [R,] log-likelihoods for each independent dimension of Y
        Y = tf.reshape(Y, (P * N, -1))  # Change 3
        log_prob = multivariate_normal(Y, m, L)
        return tf.reduce_sum(log_prob)

    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        r"""
        This method computes predictions at X \in R^{N \x D} input points

        .. math::
            p(F* | Y)

        where F* are points on the GP at new data points, Y are noisy observations at training data points.
        """
        X_data, Y_data = self.data

        P = X_data.shape[1]
        N = X_data.shape[0]
        M = Xnew.shape[0]
        R = Xnew.shape[1]

        dim = N * P
        dim_new_data = M * R

        # Change
        X_data_t = tf.repeat(X_data, P, axis=0)
        X_data_t = tf.reshape(X_data_t, (dim, -1))
        Y_data_t = tf.reshape(Y_data, (dim, -1))

        err = Y_data_t - self.mean_function(X_data_t)

        kmm = self.kernel(X_data)
        knn = self.kernel(Xnew, full_cov=full_cov)
        kmn = self.kernel(X_data, Xnew, full_cov=True)  # Change

        # Change
        kmm = tf.reshape(kmm, (dim, dim))
        knn = tf.reshape(knn, (dim_new_data, dim_new_data))
        kmn = tf.reshape(kmn, (dim, dim_new_data))

        num_data = P * N  # Change
        s = tf.linalg.diag(tf.fill([num_data], self.likelihood.variance))

        conditional = gpflow.conditionals.base_conditional
        f_mean_zero, f_var = conditional(
            kmn, kmm + s, knn, err, full_cov=True, white=False
        )  # [N, P], [N, P] or [P, N, N]

        Xnew_t = tf.repeat(Xnew, R, axis=0)  # Change 4

        f_mean = f_mean_zero + self.mean_function(Xnew_t)
        f_var = tf.linalg.diag_part(tf.squeeze(f_var, axis=0))  # Taking diagonals
        f_var = tf.reshape(f_var, (-1, R))
        return tf.transpose(f_mean), f_var
