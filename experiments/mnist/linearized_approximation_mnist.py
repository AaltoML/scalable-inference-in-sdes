import tensorflow as tf

import sys
sys.path.append("../../src/")
from sde_tf.sde_approx.linearized_approximation_general import LinearizedApproximationGeneral


class LinearizedApproximationMNIST(LinearizedApproximationGeneral):
    def __init__(self, model, latent_dim):
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
        super().__init__(model, latent_dim)
        self.input_dim = latent_dim

    def sde_drift_diffusion(self, x):
        """Compute the drift, diffusion and at x.
        Inputs
        ------
            x: tf.Tensor,
                Input data to drift and diffusion models.

        Outputs:
            mean: tf.Tensor,
                Drift at x
            L_new: tf.Tensor,
                Diffusion at x.
        """
        mean, L = self.model.predict_mean_var(x, cast_to_numpy=False)
        eyes = tf.stack([tf.eye(self.latent_dim, dtype=tf.float64)]*x.shape[0], axis=0)
        L_new = tf.linalg.set_diag(eyes, L)
        L_new = tf.math.sqrt(L_new)
        return mean, L_new

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
                concatenated in a flat format
        Outputs
        -------
            tf.Tensor,
                New drift, diffusion
        """
        batch_size = x.shape[0]
        m = x[:, :self.latent_dim]
        P = tf.reshape(x[:, self.latent_dim:], (-1, self.latent_dim, self.latent_dim))
        Q = tf.stack([tf.eye(self.latent_dim, dtype=tf.float64)]*batch_size)

        new_mean, l_val = self.sde_drift_diffusion(m)

        jac_f = self.jacobian_f_batch(m)
        jac_f = tf.reshape(jac_f, (batch_size, self.latent_dim, self.latent_dim))

        term_1 = tf.matmul(P, tf.transpose(jac_f, (0, 2, 1)))
        term_2 = tf.matmul(jac_f, P)
        term_3 = tf.matmul(l_val, Q)
        term_3 = tf.matmul(term_3, tf.transpose(l_val, (0, 2, 1)))
        S = term_1 + term_2 + term_3

        new_var = tf.reshape(S, (batch_size, -1))
        new_mean = tf.reshape(new_mean, (batch_size, -1))

        return tf.concat([new_mean, new_var], axis=1)
