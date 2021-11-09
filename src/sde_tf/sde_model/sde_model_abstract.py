"""Abstract class for SDE models to be used in SDE approximations."""

import tensorflow as tf

class SDEModel(tf.Module):
    """Abstract class for SDEModel."""

    def __init__(self):
        super(SDEModel, self).__init__()

    def predict_mean_var(self, x):
        """Predict drift and diffusion."""
        raise NotImplementedError

    def jacobian(self, x):
        """Compute drift Jacobian at x."""
        raise NotImplementedError
