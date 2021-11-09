import tensorflow as tf
from tensorflow.keras import layers
from log_file import LogFile
from src.sde_tf.sde_model import SDEModel

class LatentNeuralSDE(SDEModel):
    """Define the drift and diffusion at given point."""

    def __init__(self, latent_dim, output_path):
        """Initialize class.

        Inputs
        ------
            latent_dim: int,
                Number of latent dimensions
            output_path: str,
                Path to the saved model file
        """
        super(LatentNeuralSDE, self).__init__()
        self.model = MuVarNN(latent_dim)

        self.latent_dim = latent_dim
        self.output_path = output_path
        self.log_file = LogFile(output_path)
        self.previous_epochs = 0
        self.create_checkpoint()

    def predict_mean_var(self, x):
        """Predict drift and diffusion.

        Inputs
        ------
            x: tf.Tensor,
                Input data to the drift
                and diffusion functions.

        Outputs
        -------
            mean: tf.Tensor,
                Drift at x
            var: tf.Tensor,
                Diffusion at x
            kl: KL divergence between
                posterior and prior models
                at x.
        """
        mean, var, kl = self.model.forward(x)
        return mean, var, kl

    def jacobian(self, x):
        """Compute the Jacobian.

        Inputs
        ------
            x: tf.Tensor,
                Input data to the drift
                function to be used
                in Jacobian computation.

        Outputs
        -------
            jacobian: tf.Tensor,
                Batch Jacobian computed at x,
                w.r.t. state and time.
        """

        with tf.GradientTape() as g:
            g.watch(x)
            y, _, _ = self.predict_mean_var(x)
        jacobian = g.batch_jacobian(y, x)
        return jacobian

    def set_previous_epoch_val(self, epoch_val):
        """Set epoch number."""
        self.previous_epochs = epoch_val

    def create_checkpoint(self):
        """Defines a checkpoint and its manager.

        Saves model subclass BaseProcessLogic
        to self.output_path.
        """
        self.ckpt = tf.train.Checkpoint(
            epoch=tf.Variable(1),  model=self.model
        )
        self.manager = tf.train.CheckpointManager(
            self.ckpt, self.output_path, max_to_keep=2
        )
        self.ckpt.restore(self.manager.latest_checkpoint)
        if self.manager.latest_checkpoint:
            self.log_file.log(f"Restored from {self.manager.latest_checkpoint}")
            self.set_previous_epoch_val(int(self.ckpt.epoch))
            print(f"Restored from {self.manager.latest_checkpoint}")
        else:
            self.log_file.log("Initializing from scratch")
            print("Initializing from scratch.")

    def save_checkpoint(self):
        """Use checkpoint manager to save model."""
        self.ckpt.epoch.assign_add(1)
        self.manager.save()



class MuVarNN(tf.Module):
    """Neural network class for inferring mean and variance differentials from current state."""
    def __init__(self, latent_dim):
        """Initialize class.

        Creates drift, diffusion and prior drift
        neural networks.

        Inputs
        ------
            latent_dim: int,
                Number of latent dimensions.
        """
        super(MuVarNN, self).__init__()
        self.latent_dim = latent_dim + 1
        self.output_dim = latent_dim


        init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        self.var_nn_list = [tf.keras.Sequential([
                        layers.Dense(30, activation='softplus', kernel_initializer=init, bias_initializer=init),
                        layers.Dense(1, kernel_initializer=init, bias_initializer=init)]) for i in range(latent_dim)]
        self.mu_nn = tf.keras.Sequential([
        layers.Dense(30, activation='softplus', kernel_initializer=init, bias_initializer=init),
        layers.Dense(latent_dim, kernel_initializer=init, bias_initializer=init)
        ])
        self.prior_nn = tf.keras.Sequential([
        layers.Dense(30, activation='softplus', kernel_initializer=init, bias_initializer=init),
        layers.Dense(latent_dim, kernel_initializer=init, bias_initializer=init)
        ])


    def forward(self, x):
        """Pass data through the drift and diffusion neural networks.

        Inputs
        ------
            x: tf.Tensor,
                Data point where to compute
                drift, diffusion and KL divergence.
                Consists of both state and time.

        Outputs
        -------
            drift: tf.Tensor,
                Posterior drift at (x, t)
            var: tf.Tensor,
                Diffusion at (x, t)
            f_logp: tf.Tensor,
                KL divergence between posterior
                and prior drift at (x, t).


        """
        vars = []
        for i in range(self.output_dim):
            x_and_t = tf.stack([x[:, i], x[:, -1]], axis=-1)
            var = self.var_nn_list[i](x_and_t)
            vars.append(var)
        var = tf.concat(vars, axis=1)
        var = tf.keras.activations.sigmoid(var)
        drift = self.mu_nn(x)

        x_and_t = tf.concat([x[:, :self.output_dim], x[:, -1:]], axis=1)
        prior_drift = self.prior_nn(x_and_t)
        u = tf.math.divide_no_nan(drift - prior_drift, var)
        f_logqp = tf.reduce_sum(.5 * (u ** 2), axis=1)

        return drift, var, f_logqp
    