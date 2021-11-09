"""A tensorflow implementation for sigma point approximations."""
import tensorflow as tf
from src.sde_tf.sde_approx import SDEApprox


class SigmaPointApprox(SDEApprox):
    """General class usable for any sigma point approximation."""

    def __init__(self, weights=None, sigma_pnts=None, model=None, latent_dim=2):
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
        super(SigmaPointApprox, self).__init__()
        self.weights = weights
        self.sigma_pnts = sigma_pnts
        self.weights = tf.convert_to_tensor(self.weights, tf.float64)
        self.sigma_pnts = tf.convert_to_tensor(self.sigma_pnts, tf.float64)
        self.model = model
        self.latent_dim = latent_dim
        self.input_dim = latent_dim + 1

    def set_model(self, model):
        """Set self.model as model."""
        self.model = model

    def sde_drift_diffusion(self, x):
        """Compute drift, diffusion and KL divergence.

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
            kl: tf.Tensor,
                KL divergence between prior
                and posterior processes at x.
        """
        x = tf.reshape(x, (-1, self.input_dim))
        m, L, kl = self.model.predict_mean_var(x)
        eyes = tf.eye(self.latent_dim, dtype=tf.float64)
        L_new = tf.linalg.set_diag(eyes, L[0])
        L_new = tf.sqrt(L_new)
        return m, L_new, kl

    @tf.function(experimental_relax_shapes=True)
    def get_nxt_data(self, t, data):
        """Vectorized version of get_nxt_data_one.

        See self.get_nxt_data_one for inputs/outputs.
        """
        return tf.vectorized_map(lambda x: self.get_nxt_data_one(t, x), data)

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

        drift, diff, _ = self.sde_drift_diffusion(proj_point)
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

    def get_nxt_data_one(self, t, data):
        """Define the ODEs of m and P.

        Inputs
        ------
            t: float,
                Time at current step
            data: tf.Tensor,
                Input data vector consisting of flattened earlier mean,
                variance and KL divergence. Assumed
                batch size = 1.

        Outputs
        -------
            tf.Tensor: A tensor with same shape as data,
                consisting of mean, variance and KL divergence.

        """
        data = data[:-1] # no need to use KL divergence.
        m = data[:self.latent_dim]
        s = tf.reshape(data[self.latent_dim:], (self.latent_dim, self.latent_dim))
        s += 10**(-6) * tf.eye(self.latent_dim, dtype=tf.float64)
        sqrt_s = tf.linalg.cholesky(s)

        Q = tf.eye(self.latent_dim, dtype=tf.float64)
        mean_repeated = tf.stack([m for _ in range(self.sigma_pnts.shape[0])], axis=1)

        projected_sigma_pnts = tf.transpose(
            mean_repeated
            + sqrt_s @ tf.transpose(self.sigma_pnts)
        )

        times = tf.ones((projected_sigma_pnts.shape[0], 1), dtype=tf.float64)*t
        projected_sigma_pnts = tf.concat([projected_sigma_pnts, times], axis=1)

        drift, _, _ = self.sde_drift_diffusion(projected_sigma_pnts)

        dm_dt = tf.reshape(self.weights, (1, -1)) @ drift

        x = tf.convert_to_tensor([tf.concat([tf.reshape(self.weights[i], [1]), projected_sigma_pnts[i], self.sigma_pnts[i]], axis=0)
             for i in range(projected_sigma_pnts.shape[0])])    # combine weights, projected sigma points and sigma points for vectorization
        dp_dt = tf.vectorized_map(lambda y: self.add_sigma_point(y, sqrt_s, Q), x)
        dp_dt = tf.math.reduce_sum(dp_dt, axis=0)

        dp_dt = tf.reshape(dp_dt, [-1])
        _, _, kl = self.sde_drift_diffusion(tf.concat([m, times[0]], axis=0))
        kl = tf.expand_dims(tf.reduce_mean(kl), axis=0)
        return tf.concat([dm_dt[0], dp_dt, kl], axis=0)



