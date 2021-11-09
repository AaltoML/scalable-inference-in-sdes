import tensorflow as tf

import sys
sys.path.append("../../src/")
from src.sde_tf.sde_approx.sigma_point_approx_tf import SigmaPointApprox


class SigmaPointApproxMNIST(SigmaPointApprox):

    def __init__(self, weights=None, sigma_pnts=None, model=None, latent_dim=16):
        """Initialize class.

        Inputs
        ------
            weights: tf.Tensor,
                A vector of weights for each sigma point
            sigma_pnts: tf.Tensor,
                Sigma points to be used
            model: SDEModel,
                An SDEModel to use for computing
                the drift and diffusion.
            latent_dim: int,
                Number of state dimensions.
        """
        super().__init__(weights, sigma_pnts, model, latent_dim)
        self.input_dim = latent_dim

    def sde_drift_diffusion(self, x):
        """Compute drift, diffusion.

        Inputs
        ------
            x: tf.Tensor,
                Tensor consisting of current state and time.
                Batch as first dimension.

        Outputs
        ------
            m: tf.Tensor,
                Drift at x.
            L_new: tf.Tensor,
                Diffusion at x
        """
        x = tf.reshape(x, (-1, self.input_dim))
        m, L = self.model.predict_mean_var(x, cast_to_numpy=False)
        eyes = tf.eye(self.latent_dim, dtype=tf.float64)
        L_new = tf.linalg.set_diag(eyes, L[0])
        L_new = tf.sqrt(L_new)
        return m, L_new

    def add_sigma_point(self, y, sqrt_s, Q):
        """Compute the term of a single sigma point in the approximation.

        Inputs
        ------
            y: tf.Tensor,
                A combined vector consisting of a weight,
                projected sigma point and sigma point.
                Used as a single argument for utility in
                vectorizing.
            sqrt_S: tf.Tensor,
                Cholesky decomposition of the variance.
            Q: tf.Tensor,
                Brownian motion diffusion.
        """
        weight = y[0]
        proj_point = y[1:self.input_dim+1]
        _e = y[-self.latent_dim:]

        drift, diff = self.sde_drift_diffusion(proj_point)
        # First term
        dp_dt_i = tf.zeros((self.latent_dim, self.latent_dim), dtype=tf.float64)
        dp_dt_i += (
                tf.reshape(drift, (self.latent_dim, 1))
                @ tf.reshape(_e, (1, self.latent_dim))
                @ tf.transpose(sqrt_s)
            )

        # Second term
        dp_dt_i += (
                sqrt_s
                @ tf.reshape(_e, (self.latent_dim, 1))
                @ (tf.reshape(drift, (1, self.latent_dim)))
            )

        # Third term terms
        dp_dt_i += diff @ Q @ tf.transpose(diff)
        dp_dt_i = dp_dt_i*weight
        return dp_dt_i

    @tf.function
    def get_nxt_data_one(self, t, data):
        """Define the ODEs of m and P.

        Inputs
        ------
            t: float,
                Time at current step
            data: tf.Tensor,
                Input data vector consisting of flattened earlier mean,
                variance. Assumed batch size = 1.

        Outputs
        -------
            tf.Tensor: A tensor with same shape as data,
                consisting of mean, variance.

        """
        m = data[:self.latent_dim]
        s = tf.reshape(data[self.latent_dim:], (self.latent_dim, self.latent_dim))
        sqrt_s = tf.linalg.cholesky(s)

        Q = tf.eye(self.latent_dim, dtype=tf.float64)

        mean_repeated = tf.stack([m for _ in range(self.sigma_pnts.shape[0])], axis=1)
        projected_sigma_pnts = tf.transpose(
            mean_repeated
            + sqrt_s @ tf.transpose(self.sigma_pnts)
        )
        drift, _ = self.sde_drift_diffusion(projected_sigma_pnts)
        dm_dt = tf.reshape(self.weights, (1, -1)) @ drift

        x = tf.convert_to_tensor([tf.concat([tf.reshape(self.weights[i], [1]), projected_sigma_pnts[i], self.sigma_pnts[i]], axis=0)
            for i in range(projected_sigma_pnts.shape[0])])    # combine weights, projected sigma points and sigma points for vectorization
        dp_dt = tf.vectorized_map(lambda y: self.add_sigma_point(y, sqrt_s, Q), x)
        dp_dt = tf.math.reduce_sum(dp_dt, axis=0)

        dp_dt = tf.reshape(dp_dt, [-1])

        return tf.concat([dm_dt[0], dp_dt], axis=0)

