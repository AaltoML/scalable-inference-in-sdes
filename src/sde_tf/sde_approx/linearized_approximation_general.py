"""Linearized approximation implemented with tensorflow."""
import numpy as np
import tensorflow as tf
from src.sde_tf.sde_approx import SDEApprox


class LinearizedApproximationGeneral(tf.Module, SDEApprox):
    """Class for linearized approximation,"""
    def __init__(self, model, latent_dim=2):
        """Initialize class.

        Inputs
        ------
            model: SDEModel,
                An SDEModel class used to predict
                drift, diffusion, KL divergence
                and the Jacobian.
            latent_dim: int,
                Number of latent dimensions.
        """
        super(LinearizedApproximationGeneral, self).__init__()
        self.model = model
        self.latent_dim = latent_dim
        self.input_dim = latent_dim + 1

    def validate_covs(self, cov):
        return np.all(np.linalg.eigvals(cov) > 0)

    def set_model(self, model):
        """Reset self.model as model."""
        self.model = model

    def sde_drift_diffusion(self, x):
        """Compute the drift, diffusion and KL divergence at x.

        Inputs
        ------
            x: tf.Tensor,
                Input data to drift and diffusion models.

        Outputs:
            mean: tf.Tensor,
                Drift at x
            L_new: tf.Tensor,
                Diffusion at x.
            kl: tf.Tensor,
                KL divergence between prior
                and posterior drift processes.
        """
        mean, L, kl = self.model.predict_mean_var(x)
        eyes = tf.stack([tf.eye(self.latent_dim, dtype=tf.float64)]*x.shape[0], axis=0)
        L_new = tf.linalg.set_diag(eyes, L)
        L_new = tf.math.sqrt(L_new)
        return mean, L_new, kl

    def jacobian_f_batch(self, x):
        """Compute Jacobian at x.

        Inputs
        ------
            x: tf.Tensor,
                Input data to be used in Jacobian
                computation


        Outputs
        -------
            jac: tf.Tensor,
                Jacobian at x, only w.r.t
                the state dimensions.
        """
        jac = self.model.jacobian(x)
        jac = jac[:, :, :-1]    # remove time derivatives, only state needed
        return jac

    @tf.function(experimental_relax_shapes=True)
    def get_nxt_data(self, t, x):
        """Get the derivatives of m and P.

        Inputs
        ------
            t: float,
                Current time on the dynamics.
            data: tf.Tensor,
                The current drift and diffusion,
                concatenated in a flat format together with
                KL divergence.

        Outputs
        -------
            tf.Tensor,
                New drift, diffusion and KL divergence.
        """
        data = x[:, :-1] # disregard kl part of data
        batch_size = data.shape[0]
        m = data[:, :self.latent_dim]
        P = tf.reshape(data[:, self.latent_dim:], (-1, self.latent_dim, self.latent_dim))
        Q = tf.stack([tf.eye(self.latent_dim, dtype=tf.float64)]*batch_size)


        times_m = tf.ones((batch_size, 1), dtype=tf.float64)*t
        m = tf.concat([m, times_m], axis=1)

        new_mean, l_val, kl = self.sde_drift_diffusion(m)

        jac_f = self.jacobian_f_batch(m)

        term_1 = tf.matmul(P, tf.transpose(jac_f, (0, 2, 1)))
        term_2 = tf.matmul(jac_f, P)
        term_3 = tf.matmul(l_val, Q)
        term_3 = tf.matmul(term_3, tf.transpose(l_val, (0, 2, 1)))
        S = term_1 + term_2 + term_3

        new_var = tf.reshape(S, (batch_size, -1))
        new_mean = tf.reshape(new_mean, (batch_size, -1))
        kl = tf.reshape(kl, (batch_size, -1))

        return tf.concat([new_mean, new_var, kl], axis=1)
