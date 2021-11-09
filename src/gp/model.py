import gpflow
import tensorflow as tf
import numpy as np

tf.keras.backend.set_floatx('float64')

import src.gp.kernel as K
from src.gp.multioutput_gpr import MultiOutputGPR


class BaseLatentGPR:
    """Base Latent GPR class"""

    def __init__(self, kernel_type, output_dim, k_lengthscale=1.0, k_variance=1.0, lr=0.01, output_path=None):
        self.model = None
        self.kernel = self.init_kernel(kernel_type, output_dim, k_lengthscale, k_variance)
        self.optimizer = tf.optimizers.Adam(learning_rate=lr)
        self.output_path = output_path

    def create_checkpoint(self):
        """Create checkpoint i.e. save model. If the model already exist than the weights are loaded"""

        self.ckpt = tf.train.Checkpoint(
            epoch=tf.Variable(1), optim=self.optimizer, model=self.model
        )
        self.manager = tf.train.CheckpointManager(
            self.ckpt, self.output_path, max_to_keep=2
        )
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            print(f"Restored from {self.manager.latest_checkpoint}")
        else:
            print("Initializing from scratch.")


    def save_checkpoint(self):
        self.ckpt.epoch.assign_add(1)
        self.manager.save()

    def init_kernel(self, kernel_type, output_dim, k_lengthscale=1.0, k_variance=1.0):
        """Initialize the kernel function"""
        if kernel_type == 'Matern':
            kernel = gpflow.kernels.Matern52(
                lengthscales=k_lengthscale, variance=k_variance
            )
        elif kernel_type == 'RBF':
            kernel = gpflow.kernels.SquaredExponential(
                lengthscales=k_lengthscale, variance=k_variance
            )

        elif kernel_type == "curlfree":
            kernel = K.CurlFreeKernel(output_dim=output_dim)

        elif kernel_type == "divergencefree":
            kernel = K.DivergenceFreeKernel(output_dim=output_dim)
        else:
            print("Invalid kernel type")
            kernel = None

        return kernel

    def predict(self, x, cast_to_numpy=True):
        """Predict the posterior mean: p(y*|x*,x,y)"""
        mean, _ = self.model.predict_y(x)
        if cast_to_numpy:
            mean = mean.numpy()
        return mean

    def predict_mean_var(self, x, full_cov=False, predict_f=False, cast_to_numpy=True):
        """Predict the mean and variance of the posterior: p(y*|x*,x,y)"""
        if predict_f:
            mean, var = self.model.predict_f(x, full_cov=full_cov)
        else:
            mean, var = self.model.predict_y(x, full_cov=full_cov)
        if cast_to_numpy:
            mean = mean.numpy()
            var = var.numpy()
        return mean, var


class LatentSVGP(BaseLatentGPR):
    """Sparse Variational Gaussian Process (SVGP) for learning the latent dynamics"""

    def __init__(self, x, output_path, n_inducing_pnts=500, lr=0.001, kernel_type='RBF', num_latent_gps=16,
                 k_lengthscale=1.0, k_variance=1.0, noise_variance=1, train_noise_variance=True):
        super().__init__(kernel_type, x.shape[-1], k_lengthscale=k_lengthscale, k_variance=k_variance, lr=lr,
                         output_path=output_path)

        inducing_pnts_idx = np.random.randint(0, x.shape[0], n_inducing_pnts)
        self.inducing_pnts_idx = inducing_pnts_idx
        inducing_pnts = x[inducing_pnts_idx]
        inducing_points = gpflow.inducing_variables.InducingPoints(inducing_pnts)

        self.model = gpflow.models.SVGP(
            kernel=self.kernel,
            inducing_variable=inducing_points,
            likelihood=gpflow.likelihoods.Gaussian(),
            num_data=x.shape[0],
            num_latent_gps=num_latent_gps, )

        self.model.likelihood.variance.assign(noise_variance)
        if not train_noise_variance:
            gpflow.utilities.set_trainable(self.model.likelihood.variance, False)

        if self.output_path is not None:
            self.create_checkpoint()
        else:
            print("Model won't be saved as no output path is provided")

        self.n_latent = num_latent_gps

    def train(self, x, y, epochs=10, minibatch_size=64):
        """Train the SVGP model using minibatching"""
        train_dataset = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(x.shape[0])
        elbo = []
        print("------------------------------------------------------------------------------------")
        print("GP Train")
        print("------------------------------------------------------------------------------------")
        for e in range(epochs):
            train_iter = iter(train_dataset.batch(minibatch_size))
            elbo_mov_avg = []

            for (x_batch, y_batch) in train_iter:
                with tf.GradientTape() as tape:
                    tape.watch(self.model.trainable_variables)
                    obj = -self.model.maximum_log_likelihood_objective(
                        (x_batch, y_batch)
                    )
                    grads = tape.gradient(obj, self.model.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(grads, self.model.trainable_variables)
                )
                elbo_mov_avg.append(obj.numpy())

            elbo.append(np.mean(elbo_mov_avg))

            print(f"Epoch {int(e + 1)} ELBO: {elbo[-1]}", end='\r')

            self.save_checkpoint()

        if epochs > 0:
            print(f"Epoch {int(epochs)} ELBO: {elbo[-1]}")
        print("------------------------------------------------------------------------------------")

        return elbo

    def jacobian(self, x, batch=False, predict_f=True):
        if not isinstance(x, tf.Tensor):
            x = tf.constant(x, dtype=tf.float64)
        if not batch:
            with tf.GradientTape(persistent=True) as g:
                g.watch(x)
                y, _ = self.model.predict_f(x)
            jacobian = g.jacobian(y, x, experimental_use_pfor=False)
        else:
            with tf.GradientTape() as g:
                g.watch(x)
                y, _ = self.predict_mean_var(x, predict_f=predict_f, cast_to_numpy=False)
            jacobian = g.batch_jacobian(y, x)

        return jacobian


class LatentGPR(BaseLatentGPR):
    """Gaussian process regression class for learning the latent dynamics"""

    def __init__(self, data, output_path=None, lr=0.001, kernel_type="RBF", k_lengthscale=1.0, k_variance=1.0,
                 noise_variance=1.0, train_noise_variance=True):
        super().__init__(kernel_type, data[1].shape[-1], k_lengthscale=k_lengthscale, k_variance=k_variance, lr=lr,
                         output_path=output_path)

        if kernel_type == "RBF" or kernel_type == 'Matern':
            self.model = gpflow.models.GPR(data, self.kernel, noise_variance=noise_variance)
        else:
            self.model = MultiOutputGPR(data, self.kernel)

        if not train_noise_variance:
            gpflow.utilities.set_trainable(self.model.likelihood.variance, False)

        if self.output_path is not None:
            self.create_checkpoint()
        else:
            print("Model won't be saved as no output path is provided")

    def train(self, epochs):
        """Train the GPR model. Maximize log likelihood."""
        accumulated_obj = []
        for e in range(epochs):
            with tf.GradientTape() as tape:
                tape.watch(self.model.trainable_variables)
                obj = -self.model.maximum_log_likelihood_objective()
                grads = tape.gradient(obj, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            accumulated_obj.append(np.mean(obj.numpy()))

            print(f"Epoch {int(e)} Obj: {accumulated_obj[-1]}", end='\r')
            if self.output_path is not None:
                self.save_checkpoint()
