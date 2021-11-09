"""Define the training of a NN + ODE approximating moments."""

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tfdiffeq import odeint
tf.keras.backend.set_floatx('float64')


class ApplyApproxSDE(tf.Module):
    """Training utility class for the SDE approximation methods."""

    def __init__(self, approx_model, latent_dim, context_dim, lr=0.01, vae=None,
                 gamma=1, start_len=3, approx_prior=None, decoder_dist=False):
        """Initialize.

        Inputs
        ------
            approx_model: SDEApprox,
                Approximator model to be trained.
            latent_dim: int,
                Number of latent state dimensions.
            context_dim: int,
                Number of latent context dimensions.
            lr: float,
                Learning rate to use in training
            vae: VAETF,
                VAE class to be trained
            gamma: float,
                Weight for the KL term in the loss function.
            start_len: int,
                Number of steps taken as input to initialization.
            approx_prior: SDEApprox,
                Approximator class for the prior process, when
                that is given separately from the posterior model.
            decoder_dist: bool,
                Boolean for if the VAE outputs a distribution
        """
        super(ApplyApproxSDE, self).__init__()
        self.approx_model = approx_model
        self.approx_prior = approx_prior
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.gamma = gamma
        self.vae = vae
        self.start_len = start_len
        self.decoder_dist = decoder_dist

        print(f'Decoder dist on: {self.decoder_dist}')

        self.latent_dim += self.context_dim

    def log_prob_diag(self, stacked):
        """Calculate log prob of data given a mean and cov of a Gaussian.

        Used only for diagonal covariance matrices.

        Inputs
        -----
            stacked: tf.Tensor,
                A stacked tensor consisting of an observation,
                a mean and a covariance (diagonal)

        Outputs
        -----
            logprob: tf.Tensor,
                The log probability of generating
                the observation from the distribution
                defined by given mean and covariance.
        """
        dim = stacked.shape[-1] // 3
        x = stacked[:dim]
        mu = stacked[dim:2*dim]
        cov = stacked[2*dim:]
        scales = tf.sqrt(cov)
        dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=scales)
        logprob = dist.log_prob(x)
        return logprob

    def nll_loss(self, y, means, cov):
        """Calculate negative log-likelihood.

        Inputs
        ------
            y: tf.Tensor,
                Observational data to be compared to,
                with shape (batch_size, seq_len, dim)
            means: tf.Tensor,
                Normal distribution means
            cov: tf.Tensor,
                Normal distribution covariances,
                given as the diagonal.

        Outputs
        ------
            -ll: tf.Tensor,
                Log prob for each sample in batch at
                all observation times, shape (batch_size, seq_len).
        """
        stacked = tf.concat([y, means, cov], axis=-1)
        vec_prob = lambda x: tf.vectorized_map(self.log_prob_diag, x)
        ll = tf.vectorized_map(vec_prob, stacked)
        return -ll

    def kl_loss(self, means, cov):
        """Calculate KL divergence to N(0, I).

        Inputs
        ------
            means: tf.Tensor,
                The mean function over batch samples and time
            cov: tf.Tensor,
                The covariance function over batch samples and time,
                given as a flattened version of the full covariance matrix.

        Outputs
        ------
            kl: tf.Tensor,
                Single value tensor, KL divergence mean over batch samples
                and time.
        """
        stacked = tf.concat([means, cov], axis=-1)
        vec_kl = lambda x: tf.vectorized_map(self.kl_loss_single, x)
        matrix_kl = tf.vectorized_map(vec_kl, stacked)
        kl = tf.reduce_mean(matrix_kl)
        return kl

    def kl_loss_single(self, x):
        """KL loss between a distribution and N(0, I).

        Used for a single batch sample and observation time.

        Inputs
        ------
            x: tf.Tensor,
                A stacked input, containing means and a flattened
                covariance matrix.

        Outputs
        -------
            loss: tf.Tensor,
                A single KL divergence value for the given mean and cov.
        """
        means = x[:self.latent_dim]
        cov = tf.reshape(x[self.latent_dim:], (self.latent_dim, self.latent_dim))

        logdet = tf.linalg.logdet(cov)
        cov_trace = tf.linalg.trace(cov)
        mean_term = tf.reduce_sum(means*means, axis=-1)

        kl = 0.5 * (-logdet - self.latent_dim  + cov_trace + mean_term)
        return kl

    def kl_div_single_general(self, x):
        """Calculate KL-divergence for single mean and covariance pair.

        Can be used for any pair of means and covariances.

        Inputs
        ------
            x: tf.Tensor,
                A stacked input consisting
                of two means and two flattened
                covariance matrices.

        Outputs
        -------
            kl_div: tf.Tensor,
                KL divergence between the
                distributions N(mean1, cov1)
                and N(mean2, cov2).

        """
        kl_dim = self.latent_dim
        mean1 = x[:kl_dim]
        mean2 = x[kl_dim:2*kl_dim]
        cov1 = x[2*kl_dim:(2*kl_dim + kl_dim**2)]
        cov2 = x[-kl_dim**2:]
        cov1 = tf.reshape(cov1, (kl_dim, kl_dim))
        cov2 = tf.reshape(cov2, (kl_dim, kl_dim))

        diag1 = tf.linalg.diag_part(cov2)
        inv2 = tf.linalg.set_diag(tf.eye(kl_dim, dtype=tf.float64), 1/diag1)

        logdet1 = tf.linalg.logdet(cov1)
        logdet2 = tf.reduce_sum(tf.math.log(diag1))

        mult_trace = tf.linalg.trace(inv2@cov1)

        mean_term = tf.reduce_sum(((mean2 - mean1)**2)*tf.linalg.diag_part(inv2))
        kl_div = 0.5*(logdet2 - logdet1 - kl_dim + mult_trace + mean_term)
        return kl_div


    def get_only_state(self, cov):
        """Extract the part of a covariance matrix which corresponds to the state dimensions.

        Removes context dimension from a flat covariance matrix.

        Inputs
        -----
            cov: tf.Tensor,
                A covariance matrix over batch samples and time.

        Outputs
        ------
            cov_state: tf.Tensor,
                A state-dimension only covariance matrix,
                in the flat format (batch_size, seq_len, state_dim**2)

        """
        batch_size = cov.shape[0]
        latent_state_dim = self.latent_dim - self.context_dim
        cov_state = tf.reshape(cov, (batch_size, -1, self.latent_dim, self.latent_dim))
        cov_state = cov_state[:, :, :latent_state_dim, :latent_state_dim]
        cov_state = tf.reshape(cov_state, (batch_size, -1, latent_state_dim**2))
        return cov_state

    def kl_vae(self, init_dist):
        """KL divergence of encoder distribution at z_0 to normal distribution.

        Inputs
        -----
            init_dist: tf.Tensor,
                Initial distribution in a flat format,
                concatenated mean and flattened covariance matrix.

        Outputs
        ------
            kl: tf.Tensor,
                KL divergence between the initial encoder distribution and
                the prior (N(0, I)).

        """
        mean_enc = tf.expand_dims(init_dist[:, :self.latent_dim], 0)
        cov_enc = tf.expand_dims(init_dist[:, self.latent_dim:], 0)
        kl = self.kl_loss(mean_enc, cov_enc)
        return kl

    def reconst_mse(self, y, mu):
        """Compute reconstrucion MSE.

        Inputs
        ------
            y: tf.Tensor,
                The true observational data to be
                compared to.
            mu: tf.Tensor,
                The mean ( or latent sample) to be decoded.
        Outputs
        -------
            mse: tf.Tensor,
                MSE of the decoded mean compared to
                observational data.

        """
        batch_size = y.shape[0]
        dim = y.shape[-1]
        latent_state_dim = self.latent_dim - self.context_dim

        mean_flat = tf.reshape(mu[:, :, :latent_state_dim], (-1, latent_state_dim))
        y_rec = self.vae.predict_decoder(mean_flat)
        y_rec = tf.reshape(y_rec, (batch_size, -1, dim))

        mse = tf.reduce_mean((y - y_rec)**2)
        return mse

    def sample_approx(self, means, covs):
        """Get a sample from the latent approximate distribution.

        Applies the reparametrization trick for a general covariance function.

        Inputs
        ------
            means: tf.Tensor,
                Mean function values over batch samples and time.
            covs: tf.Tensor,
                Covariance function values over batch sample and time,
                expected to be in shape (batch_size, seq_len, latent_dim^2)

        Outputs
        ------
            z: tf.Tensor,
                Samples from the distribution N(means, covs),
                with a single sample per time and batch sample.

        """
        cov = tf.reshape(covs, (*covs.shape[:2], self.latent_dim, self.latent_dim))
        sqrt_cov = tf.linalg.cholesky(cov)
        z = tf.random.normal(means.shape, dtype=tf.float64)
        z = means + tf.linalg.matvec(sqrt_cov, z)
        return z

    def validate_cov(self, covs):
        """Validate the covariance matrices.

        Checks if the covariance matrices given by the approximation
        are positive-definite.

        Inputs
        -----
            covs: tf.Tensor,
                Covariance function values over batch samples and time.
                Expected to be the shape (batch_size, seq_len, self.latent_dim**2)
        """
        batch_size = covs.shape[0]
        seq_len = covs.shape[1]
        covs = tf.reshape(covs, (batch_size, seq_len, self.latent_dim, self.latent_dim))
        batch_size = covs.shape[0]
        seq_len = covs.shape[1]
        for i in range(batch_size):
            for j in range(seq_len):
                eigens = tf.eigvals(covs[i, j]).numpy()
                assert np.all(eigens > 0), f'Not positive definite: {eigens} and {(i, j)}, {covs[i, j]}.'

    def reconst_nll(self, y, z):
        """Non negative likelihood for the decoder distribution.

        Reconstruct the decoder distribution based on
         the latent sample/distribution and evaluate
        the NLL of the observational data.

        Inputs
        ------
            y: tf.Tensor,
                Observational data.
            z: tf.Tensor,
                Latent samples to decode. Can be a sample
                from the latent Gaussian distribution, or
                a mean of the latent distribution.

        Outputs
        ------
            nll: tf.Tensor,
                Mean NLL over samples and sum/mean over times.
        """
        batch_size = y.shape[0]
        dim = y.shape[-1]
        latent_state_dim = self.latent_dim - self.context_dim

        mean_flat = tf.reshape(z[:, :, :latent_state_dim], (-1, latent_state_dim))
        y_rec_mu, y_rec_logvar = self.vae.decoder.forward(mean_flat)
        y_rec_mu = tf.reshape(y_rec_mu, (batch_size, -1, dim))
        y_rec_var = tf.math.exp(y_rec_logvar)
        y_rec_var = tf.reshape(y_rec_var, (y.shape[0], -1, dim))

        nll = self.nll_loss(y, y_rec_mu, y_rec_var)

        nll = tf.reduce_mean(nll)
        return nll

    def vae_init_loss(self, y, init_dist):
        """Initial MSE loss of the VAE reconstruction at t=0.

        Reconstruct observation number self.start_len from the encoded
        distribution and evaluate it negative log likelihood.

        Inputs
        -------
            y: tf.Tensor,
                Observational data. Only the observation number
                self.start_len will be used.
            init_dist: tf.Tensor,
                Initial encoder distribution. Consists of a stacked
                mean and flattened covariance matrix.

        Outputs
        -------
            loss: tf.Tensor,
                Initial VAE loss. Either MSE or NLL, depending on the
                type of decoder used.
        """
        latent_state_dim = self.latent_dim - self.context_dim
        init_point = y[:, self.start_len - 1]
        init_mu = init_dist[:, :latent_state_dim]
        init_var = init_dist[:, self.latent_dim:]
        init_var = self.get_only_state(init_var)[:, 0, :] # use only initial var
        init_var = tf.reshape(init_var, (-1, latent_state_dim, latent_state_dim))
        init_var = tf.linalg.diag_part(init_var)
        init_logvar = tf.math.log(init_var)

        latent_sample = self.vae.encoder.draw_sample(init_mu, init_logvar)
        if self.decoder_dist:
            reconst_init, reconst_init_logvar = self.vae.decoder.forward(latent_sample)
            reconst_init_var = tf.math.exp(reconst_init_logvar)
            reconst_mu = tf.expand_dims(reconst_init, 1)
            reconst_init_var = tf.expand_dims(reconst_init_var, 1)
            init_point_2 = tf.expand_dims(init_point, 1)
            loss = self.nll_loss(init_point_2, reconst_mu, reconst_init_var)
            loss = tf.reduce_mean(loss)
        else:
            reconst_init = self.vae.predict_decoder(latent_sample)
            loss = self.eval_mse(reconst_init, init_point)
        return loss

    def prior_fixed_loss(self, y, means, cov, init_dist, prior_means, prior_covs):
        """Compute loss function when prior is a fixed process.

        Inputs
        -------
            y: tf.Tensor,
                Observational data, expected to have shape
                (batch_size, seq_len, n_dim)
            means: tf.Tensor,
                Latent mean function trajectory values
            cov: tf.Tensor,
                Latent covariance function trajectory values.
            init_dist: tf.Tensor,
                The initial encoded position, based on the first
                self.start_len points in y.
            prior_means: tf.Tensor,
                Latent prior mean function trajectory values
            prior_covs: tf.Tensor,
                Latent prior covariance function trajectory values.

        Outputs
        -------
            loss: tf.Tensor,
                Averaged loss. Consists of a KL term,
                reconstruction loss and initial point
                reconstruction loss.
            mse: tf.Tensor,
                MSE between observational data and decoded
                latent data (samples or mean)

        """

        y_traj = y[:, self.start_len:]
        latent_vals = self.sample_approx(means, cov)
        # nll
        if self.decoder_dist:
            recon_nll = self.reconst_nll(y_traj, latent_vals)
        else:
            recon_nll = self.reconst_mse(y_traj, latent_vals)
        # trajectory kl
        model_kl = self.kl_loss_general(means, cov, prior_means, prior_covs, mean=True)

        # kl from encoding
        encoder_kl = self.kl_vae(init_dist)

        # initial decoder loss
        vae_init_loss = self.vae_init_loss(y, init_dist)

        weight = self.gamma*tf.convert_to_tensor(min(self.epoch*0.005, 1), tf.float64)
        loss = recon_nll + encoder_kl*self.gamma + weight**model_kl + vae_init_loss

        mse = self.reconst_mse(y_traj, latent_vals)
        return loss, mse

    def prior_fit_loss(self, y, means, cov, init_dist, kl):
        """Compute the loss when using fit prior drift.

        Compute the loss when the prior has a matching diffusion but
        fit drift. Expected to have KL divergence as model output.

        Inputs
        ------
            y: tf.Tensor,
                Observational data, expected to have shape
                (batch_size, seq_len, n_dim)
            means: tf.Tensor,
                Latent mean function trajectory values
            cov: tf.Tensor,
                Latent covariance function trajectory values.
            init_dist: tf.Tensor,
                The initial encoded position, based on the first
                self.start_len points in y.
            kl: tf.Tensor,
                KL divergence values over the trajectory.

        Outputs
        -------
            loss: tf.Tensor,
                Averaged loss. Consists of a KL term,
                reconstruction loss and initial point
                reconstruction loss.
            mse: tf.Tensor,
                MSE between observational data and decoded
                latent data (samples or mean)
        """
        y_traj = y[:, self.start_len:]

        latent_vals = self.sample_approx(means, cov)

        if self.decoder_dist:
            recon_nll = self.reconst_nll(y_traj, latent_vals)
        else:
            recon_nll = self.reconst_mse(y_traj, latent_vals)
        vae_init_loss = self.vae_init_loss(y, init_dist)
        encoder_kl = self.kl_vae(init_dist)

        if self.mean_of_traj:
            kl = tf.reduce_mean(kl)
        else:
            kl = tf.reduce_mean(tf.reduce_sum(kl, axis=1))
        weight = self.gamma*tf.convert_to_tensor(min(self.epoch*0.005, 1), tf.float64)

        loss = kl*weight + recon_nll + encoder_kl*self.gamma + vae_init_loss
        mse = self.reconst_mse(y_traj, means)
        return loss, mse

    def kl_loss_general(self, mean1, cov1, mean2, cov2, mean=True):
        """KL-divergence for two multivariate Gaussian distributions.

        Covers the case where the covariance matrices are not
        diagonal!
        """
        stacked = tf.concat([mean1, mean2, cov1, cov2], axis=-1)
        vec_kl = lambda x: tf.vectorized_map(self.kl_div_single_general, x)
        matrix_kl = tf.vectorized_map(vec_kl, stacked)
        if mean:
            kl = tf.reduce_mean(matrix_kl)
        else:
            kl = tf.reduce_mean(tf.reduce_sum(matrix_kl, axis=1))
        return kl

    def eval_mse(self, y, y_hat):
        """Simple MSE function."""
        return tf.reduce_mean((y - y_hat)**2)

    def get_init_dist_nn(self, y, prior=False):
        """Create an initial point for the ODE.

        This is the version matching the LatentSDE paper:
        encode 3 frames to get encoder distribution for state,
        and encoder distribution for context.
        """
        batch_size = y.shape[0]
        first_n = tf.reshape(y[:, :self.start_len], (batch_size, -1))
        latent_mu, latent_var  = self.vae.encoder.forward(first_n)
        latent_var = tf.math.exp(latent_var)
        init_var = tf.stack([tf.eye(self.latent_dim, dtype=tf.float64)]*batch_size, axis=0)
        init_var = tf.linalg.set_diag(init_var, latent_var)

        var_shape = self.latent_dim
        if prior:
            var_shape = self.latent_dim - self.context_dim
            latent_mu = latent_mu[:, :var_shape]
            init_var = init_var[:, :var_shape, :var_shape]

        init_var = tf.reshape(init_var, (batch_size, var_shape**2))
        init_state = tf.concat([latent_mu, init_var], axis=1)
        return init_state

    def init_models(self, ys):
        """Initialize VAE and latent model.

        If given a prior model, initialize that too.

        Inputs
        ------
            ys: tf.Tensor,
                Training data to be used for initialization.
        """
        nn_init = self.get_init_dist_nn(ys, prior=False)

        nn_init = nn_init[:, :self.latent_dim]
        nn_init = tf.concat([nn_init, 0.1*tf.ones((nn_init.shape[0], 1), dtype=tf.float64)], axis=-1)
        mu, var, kl = self.approx_model.model.predict_mean_var(nn_init)
        latent_state_dim = self.latent_dim - self.context_dim
        mu_to_decode = mu[:, :latent_state_dim]
        _ = self.vae.predict_decoder(mu_to_decode)

    def run_trajectory(self, init_state, eval_times, ode_fn):
        """Run ODE solver.

        Inputs
        ------
            init_state: tf.Tensor,
                Initial latent state data, consisting
                of a combination of mean and flattened variance
                from the encoder
            eval_times: tf.Tensor,
                A vector of times where to evaluate the state of the dynamics
            ode_fn: function,
                A function which gives the ODE dynamics for mean and variance.
                Often the SDEApprox get_nxt_data function, or a lambda function
                derived from it.

        Outputs
        ------
            means: tf.Tensor,
                The ODE mean trajectory at eval_times.
            covs: tf.Tensor,
                The ODE variance trajectory at eval_times
            kl: tf.Tensor,
                The KL divergence between the prior and posterior
                processes at eval_times.

        """
        states_and_kl = odeint(ode_fn, init_state, eval_times, rtol=1e-7, atol=1e-8, method='dopri5')
        states = states_and_kl[1:, :, :-1]
        kl = states_and_kl[1:, :, -1]
        means = tf.transpose(states[:, :, :self.latent_dim], (1, 0, 2))
        covs = tf.transpose(states[:, :, self.latent_dim:], (1, 0, 2))
        kl = tf.transpose(kl, (1, 0))
        self.validate_cov(covs)
        return means, covs, kl

    def get_moment_trajectory(self, y, eval_times, prior=False):
        """Get mean and variance trajectories.

        Define the function to be used in the ODE solver,
        get initial data, and start the run.

        Inputs
        ------
            y: tf.Tensor,
                Input data, the full trajectory. To be used only
                for initialization.
            eval_times: tf.Tensor,
                A vector of times where to evaluate the state of the dynamics.
            prior: bool,
                Boolean for if the model to be used is a separate prior model.

        Outputs
        ------
            init_latent_state: tf.Tensor,
                Initial state without the KL placeholder term
            means: tf.Tensor,
                The ODE mean trajectory at eval_times.
            covs: tf.Tensor,
                The ODE variance trajectory at eval_times.
            kl: tf.Tensor,
                KL divergence between posterior and prior
                processes.

        """
        init_latent_state = self.get_init_dist_nn(y, prior=False)
        init_kl = tf.zeros((y.shape[0], 1), dtype=tf.float64)
        init_latent_state = tf.concat([init_latent_state, init_kl], axis=1)
        if prior:
            ode_fn = self.approx_prior.get_nxt_data
        else:
            ode_fn = self.approx_model.get_nxt_data
        means, covs, kl = self.run_trajectory(init_latent_state, eval_times, ode_fn)
        return init_latent_state[:, :-1], means, covs, kl

    def train_step(self, y, eval_times):
        """Run single training step.

        Run posterior (and possibly prior) trajectories,
        compute the loss and take an optimizer step.

        Inputs
        -------
            y: tf.Tensor,
                Tensor containing observational data to be
                modelled
            eval_times: tf.Tensor,
                A list of observation times where to evaluate the state
                of the dynamics.

        Outputs
        ------
            mse: tf.Tensor,
                MSE of the model in the observation space
            loss: tf.Tensor,
                Loss function values.
        """
        with tf.GradientTape() as tape:
            tape.watch(self.approx_model.model.trainable_variables + self.vae.trainable_variables)
            init_latent, means, covs, kl = self.get_moment_trajectory(
                                        y, eval_times, prior=False)
            if self.run_prior_trajectory:
                prior_init, prior_means, prior_covs, _ = self.get_moment_trajectory(
                            y, eval_times, prior=True)
                loss, mse = self.prior_fixed_loss(y,  means, covs, init_latent, prior_means, prior_covs)
            else:
                loss, mse = self.prior_fit_loss(y,  means, covs, init_latent, kl)
        grads = tape.gradient(loss, self.approx_model.model.trainable_variables \
                                            + self.vae.trainable_variables)
        assert None not in grads, f'Some gradients were not defined! {grads}'
        self.optimizer.apply_gradients(
                        zip(grads,  self.approx_model.model.model.trainable_variables \
                                    + self.vae.trainable_variables))
        return mse, loss

    def train_full(self,  ys, val_ys, epochs=10, dt=0.1, minibatch_size=16):
        """Train the VAE + Latent Model.

        Inputs
        ------
            ys: tf.Tensor,
                Training data, from the observation space.
            val_ys: tf.Tensor,
                Validation data, to be evaluated to avoid overfitting.
            epochs: int,
                Number of training epochs
            dt: float,
                Time step length between observations.
            minibatch_size: int,
                Number of samples in minibatch.

        Outputs
        ------
            SDEModel: A trained latent model.
        """
        init_time = 0.1
        end_time = dt*ys.shape[1] - self.start_len*dt
        eval_times = tf.convert_to_tensor(np.arange(init_time, end_time + 2*dt, dt), dtype=tf.float64)

        dataset = tf.data.Dataset.from_tensor_slices(ys).shuffle(ys.shape[0])
        self.init_models(ys)
        self.run_prior_trajectory = self.approx_prior is not None
        for epoch in range(epochs):
            train_iter = iter(dataset.batch(minibatch_size))
            self.epoch = epoch
            total_loss = []
            mse_losses = []
            for y in train_iter:
                mse, loss = self.train_step(y, eval_times)
                mse_losses.append(mse.numpy())
                total_loss.append(loss.numpy())
            print(f'Epoch {epoch} completed with loss {np.mean(total_loss)}')
            print(f' and with observation space MSE of {np.mean(mse_losses)}')
            if epoch % 10 == 0:
                init_latent_val, means_val, covs_val, kl_val = self.get_moment_trajectory(
                         val_ys, eval_times, prior=False)
                y_traj_val = val_ys[:, self.start_len:]
                mse_val = self.reconst_mse(y_traj_val, means_val)
                print(f'Epoch {epoch} completed with validation MSE of {mse_val}')
            self.approx_model.model.save_checkpoint()
            self.vae.save_checkpoint()
            if self.approx_prior is not None:
                self.approx_prior.model.save_checkpoint()
        return self.approx_model.model

    def generate(self, data, latent_model, dt):
        """Generate mean and variance ODE trajectories.


        Inputs
        ------
            data: tf.Tensor,
                The dataset to approximate, with shape
                (batch_size, seq_len, n_dim), only the
                first self.start_len elements are used
                for initialization.
            latent_model: SDEModel,
                A trained latent model to be applied
                in the approximation.
            dt: float,
                Time step length between observations.

        Outputs
        ------
            means: tf.Tensor,
                Generated mean ODE trajectory.
            covs: tf.Tensor,
                Generated variance ODE trajectory.
        """
        self.approx_model.set_model(latent_model)
        end_time = (data.shape[1] - self.start_len)*dt

        eval_times = tf.convert_to_tensor(np.arange(0.1, end_time + 2*dt, dt), tf.float64)
        init_state, means, covs, kl = self.get_moment_trajectory(data, eval_times,  prior=False)
        covs = self.get_only_state(covs)
        means = means[:, :, :self.latent_dim - self.context_dim]
        return means, covs
