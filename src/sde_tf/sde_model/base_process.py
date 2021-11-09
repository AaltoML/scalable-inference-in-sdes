"""Define a base process to use as a prior.

Leads to a prior which drives for positive covariance.
"""

import tensorflow as tf
from log_file import LogFile
from src.sde_tf.sde_model import SDEModel


class BaseProcess(SDEModel):
    """Model for a simple prior process.

    Utility wrapper for the process definition
    of BaseProcessLogic.
    """

    def __init__(self, latent_dim, output_path):
        """Initialize module.

        Inputs
        ------
            latent_dim: int,
                Number of latent dimensions.
            output_path: str,
                Path to the saved model file.
        """
        super(BaseProcess, self).__init__()
        self.model = BaseProcessLogic(latent_dim)

        self.latent_dim = latent_dim
        self.previous_epochs = 0
        self.output_path = output_path
        self.log_file = LogFile(output_path)
        self.create_checkpoint()

    def predict_mean_var(self, x):
        """Predict mean and variance.

        Inputs
        ------
            x: tf.Tensor,
                Data to be used in the drift
                and diffusion functions. Only
                its shape is utilized here.

        Outputs
        ------
            mean: tf.Tensor,
                Drift at the given point
            var: tf.Tensor,
                Diffusion at the given point
            kl: tf.Tensor,
                Placeholder for Kl divergence,
                passed here since it is expected
                as output.
        """
        mean, var, kl = self.model.forward(x)
        return mean, var, kl

    def jacobian(self, x):
        """Jacobian of the prior.

        Inputs
        ------
            x: tf.Tensor,
                Data to be used in the Jacobian calculation.
                Only its shape is utilized here.

        Outputs
        ------
            jacobian: tf.Tensor,

        """
        jacobian = tf.zeros((x.shape[0], self.latent_dim, self.latent_dim + 1), dtype=tf.float64)
        return jacobian

    def set_previous_epoch_val(self, epoch_val):
        """Set value of epochs."""
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


class BaseProcessLogic(tf.Module):
    """Define a white noise driven process.

    Always has drift f(x, t) = 0, diffusion L(x, t) = sigma*I.
    """
    def __init__(self, latent_dim):
        """Initialize class.

        Inputs
        -----
            latent_dim: int,
                Number of dimensions.
        """
        super(BaseProcessLogic, self).__init__()
        self.latent_dim = latent_dim
        self.sigma = tf.convert_to_tensor(0.1, dtype=tf.float64)

    def forward(self, x):
        """Predict drift, diffusion, and 'kl divergence'.

        Inputs
        -----
            x: tf.Tensor,
                Input data, only used for shape

        Outputs
        ------
            mu: tf.Tensor,
                Drift at x
            var: tf.Tensor,
                Diffusion at x
            kl: tf.Tensor,
                Placeholder for kl,
                used here since it is needed
                upstream.
        """
        batch_size = x.shape[0]
        mu = tf.zeros((batch_size, self.latent_dim), dtype=tf.float64)
        var = self.sigma*tf.ones((batch_size, self.latent_dim), dtype=tf.float64)
        kl = tf.zeros(batch_size, dtype=tf.float64)
        return mu, var, kl