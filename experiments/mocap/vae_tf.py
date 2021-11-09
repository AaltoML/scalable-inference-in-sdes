"""Tensorflow implementation of VAE, for end to end model."""

import tensorflow as tf
import tensorflow.keras.layers as layers
from log_file import LogFile
tf.keras.backend.set_floatx('float64')



class Encoder(tf.Module):
    def __init__(self, latent_dim=16):
        """Initialize encoder.

        Inputs
        -----
            latent_dim: int,
                Number of latent dimensions to encode to.
        """
        super(Encoder, self).__init__()

        init = tf.initializers.GlorotNormal()
        self.min_logvar = tf.ones(latent_dim, dtype=tf.float64)*(-6)
        self.encoder_nn = tf.keras.Sequential([
                layers.Dense(30, activation='softplus', kernel_initializer=init),
                layers.Dense(30, activation='softplus', kernel_initializer=init)]
            )

        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(latent_dim, activation='linear', kernel_initializer=init, dtype=tf.float64)
        self.fc2 = layers.Dense(latent_dim, activation='linear', kernel_initializer=init, dtype=tf.float64)

    def forward(self, x):
        """Encode data.

        Inputs
        -----
            x: tf.Tensor,
                Data to be encoded.

        Outputs
        ------
            mean: tf.Tensor,
                Encoder distribution mean
            log_var: tf.Tensor,
                Logarithm of encoder distribution variance.
        """
        x = self.encoder_nn(x)
        mean = self.fc1(x)
        log_var = self.fc2(x)
        log_var = self.min_logvar + tf.nn.relu(log_var - self.min_logvar)
        return mean, log_var

    def draw_sample(self, z_mean, z_logvar):
        """Draw a single sample from encoder distribution.

        Inputs
        ------
            z_mean: tf.Tensor,
                Distribution mean
            z_logvar:
                Logarithm of distribution variance.

        Outputs
        ------
            sample: tf.Tensor,
                A sample from the distribution.
        """
        sample = tf.random.normal(z_mean.shape, dtype=tf.float64)
        sample = z_mean + sample * (tf.math.exp(0.5 * z_logvar))
        return sample


class Decoder(tf.Module):
    def __init__(
        self,  obs_dim=50, decoder_dist=False):
        """Initialize decoder.

        Inputs
        ------
            obs_dim: int,
                Number of dimensions in observation space.
            decoder_dist: bool,
                Boolean for if the decoder should outout a distribution.
        """
        super(Decoder, self).__init__()

        init = tf.initializers.GlorotNormal()
        self.first_layers = tf.keras.Sequential([
                layers.Dense(30, activation='softplus', kernel_initializer=init),
                layers.Dense(30, activation='softplus', kernel_initializer=init)])
        self.decoder_dist = decoder_dist
        self.min_logvar = tf.ones(obs_dim, dtype=tf.float64)*(-6)
        if self.decoder_dist:
            self.mu_layer = layers.Dense(obs_dim, activation='linear',
                                         dtype=tf.float64, kernel_initializer=init)
            self.logvar_layer = layers.Dense(obs_dim, activation='sigmoid', dtype=tf.float64,
                                             kernel_initializer=init)
        else:
            self.final_layer = layers.Dense(obs_dim, activation='linear',
                                         dtype=tf.float64, kernel_initializer=init)


    def forward(self, x):
        """Compute decoder output.

        Inputs
        ------
            x: tf.Tensor,
                Latent data to be decoded.

        Outputs
        ------
            x: tf.Tensor,
                Decoded point in observation space
                (when no decoder distribution is used)
            mu: tf.Tensor,
                Mean of decoder distribution.
            logvar: tf.Tensor:
                Log of decoder distribution variance.
        """
        x = self.first_layers(x)
        if self.decoder_dist:
            mu = self.mu_layer(x)
            logvar = self.logvar_layer(x)
            logvar = self.min_logvar + tf.nn.relu(logvar - self.min_logvar)
            return mu, logvar
        else:
            x = self.final_layer(x)
        return x

    def draw_sample(self, mu, logvar):
        """Draw sample from decoder distribution.

        Inputs
        ------
            mu: tf.Tensor,
                Mean to be used in sampling.
            logvar: tf.Tensor,
                Log of variance diagonal to be
                used in sampling.

        Outputs
        ------
            sample: tf.Tensor,
                A single sample from the decoder distribution.
        """
        sample = tf.random.normal(mu.shape, dtype=tf.float64)
        sample = mu + sample * (tf.math.exp(0.5 * logvar))
        return sample


class VAETF(tf.Module):
    def __init__(self, latent_dim=16, obs_dim=50,
                 context_shape=None, encoder_path=None, decoder_path=None,
                 decoder_dist=False):
        """Initialize VAE.

        Inputs
        ------
            latent_dim: int,
                Number of latent dimensions
            obs_dim: int,
                Number of dimensions in observation space
            decoder_shape: int,
                Optional resetting of decoder
            context_shape: int,
                If givem, assumed to have context on.
            encoder_path: str,
                Path to saved/to be saved encoder model.
            decoder_path: str,
                Path of saved/to be saved decoder model.
            decoder_dist: bool,
                Boolean for if the decoder output is a distribution
                instead of a single point in observation space.

        """
        super(VAETF, self).__init__()  # It was convenient for me to have this be a torch.nn module, does this break
                                    # something in other experiments?

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(obs_dim=obs_dim,
                               decoder_dist=decoder_dist)
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.decoder_dist = decoder_dist

        self.encoder_path = encoder_path
        self.previous_epochs = 0

        self.log_file_dec = LogFile(decoder_path)
        self.log_file_enc = LogFile(encoder_path)
        self.decoder_path = decoder_path
        self.create_checkpoints()

    def create_checkpoints(self):
        """Create checkpoint files and managers for encoder and decoder."""
        enc_ckpt, enc_manager = self.create_checkpoint(self.encoder, self.encoder_path, self.log_file_enc)
        dec_ckpt, dec_manager = self.create_checkpoint(self.decoder, self.decoder_path, self.log_file_dec)
        self.enc_ckpt = enc_ckpt
        self.dec_ckpt = dec_ckpt
        self.enc_manager = enc_manager
        self.dec_manager = dec_manager

    def create_checkpoint(self, obj, path, log_file):
        """Create a checkpoint file for an object.

        Inputs
        ------
            obj: tf.Module,
                Class to save
            path: str,
                Path to model saved file
            log_file: LogFile,
                Log file object.
        """
        ckpt = tf.train.Checkpoint(
            epoch=tf.Variable(1),  model=obj
        )
        manager = tf.train.CheckpointManager(
            ckpt, path, max_to_keep=2
        )
        ckpt.restore(manager.latest_checkpoint)
        if manager.latest_checkpoint:
            log_file.log(f"Restored from {manager.latest_checkpoint}")
            self.set_previous_epoch_val(int(ckpt.epoch))
            print(f"Restored from {manager.latest_checkpoint}")
        else:
            log_file.log("Initializing from scratch")
            print("Initializing from scratch.")
        return ckpt, manager

    def set_previous_epoch_val(self, epoch_val):
        """Set previous epoch val to epoch val"""
        self.previous_epochs = epoch_val

    def save_checkpoint(self):
        """Save encoder and decoder state to file."""
        self.dec_ckpt.epoch.assign_add(1)
        self.enc_ckpt.epoch.assign_add(1)
        self.dec_manager.save()
        self.enc_manager.save()

    def predict_decoder(self, z):
        """Decode data.

        If decoded as a distribution, sample said distribution.

        Inputs
        ------
            z: tf.Tensor,
                Latent data to decode.

        Outputs
        ------
            y: tf.Tensor,
                Observation space decoded data,
                either decoded to point or sampled from decoded distribution.
        """
        if self.decoder_dist:
            mu, logvar = self.decoder.forward(z)
            y = self.decoder.draw_sample(mu, logvar)
        else:
            y = self.decoder.forward(z)
        return y
