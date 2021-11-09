"""A collection of utility functions for the MOCAP experiment."""
import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import scipy.stats

from vae_tf import VAETF
from src.sde_tf.sde_approx import (
    SigmaPointApprox,
    LinearizedApproximationGeneral,
    get_unscented_sigma_points,
    get_unscented_weights,
    get_cubature_sigma_points,
    get_cubature_weights)
from src.sde_tf.sde_model import LatentNeuralSDE, BaseProcess
from dataset_walk_tf import CMUWalkingTF
from train_approx_ode import ApplyApproxSDE


def get_vae(
        latent_dim, obs_dim,  context_dim, encoder_path=None, decoder_path=None,
        decoder_dist=False):
    """Get tensorflow VAE model.

    Inputs
    ------
        latent_dim: int,
            Number of latent positional dimensions
        obs_dim: list of int,
            The dimension of observations, as a list.
        context_dim: int,
            Number of latent context dimensions
        encoder_path: str,
            Path to encoder model checkpoint
        decoder_path: str,
            Path to decoder model checkpoint
        decoder_dist: bool,
            Boolean for if the latent samples are decoded to a single point (False) or
            to a mean and variance (True)

    Outputs:
        vae: VAETF,
            Initialized VAE tensorflow model.
    """
    latent_dim_enc = latent_dim + context_dim
    vae = VAETF(
            obs_dim=obs_dim, latent_dim=latent_dim_enc, encoder_path=encoder_path,
            decoder_path=decoder_path, decoder_dist=decoder_dist)
    return vae

def get_latent_model(output_path, latent_dim, context_dim, prior=False):
    """Get a latent model.


    Inputs
    -----
        output_path: str,
            Path to SDE model checkpoint
        latent_dim: int,
            Number of latent positional dimensions
        context_dim: int,
            Number of latent context dimensions
        prior: bool,
            Boolean for if the model initialized is a prior model

    Outputs
    ------
        latent_model:

    """
    latent_nn_dim = latent_dim + context_dim

    if not prior:
        latent_model = LatentNeuralSDE(latent_dim=latent_nn_dim,
                                   output_path=output_path)
    elif 'fit' not in output_path:
        latent_model = BaseProcess(latent_nn_dim,
                                   output_path=output_path)
    else:
        latent_model = None
    return latent_model


def eval_test_performance(base_folder, task, mean_traj, var_traj,  vae, y_true, latent_dim, start_len=3):
    """Evaluate test performance of the model.

    Inputs
    ------
        base_folder: str,
            Path to be used as base for saving plots.
        task: str,
            Name of the task, to be included in plot name.
        mean_traj: tf.Tensor,
            ODE trajectory of mean
        var_traj: tf.Tensor,
            ODE trajectory of variance
        vae: VAETF,
            Trained VAE to be used for decoding.
        y_true: tf.Tensor,
            Actual observational data to be compared to.
        latent_dim: int,
            Number of latent dimensions
        start_len: int,
            Length of starting sequence to encode for initial
            position.
    """
    mean_decoded = latent_to_obs(mean_traj, vae)
    print(f'MSE of decoded latent mean trajectory: {tf.reduce_mean((y_true[:, start_len:, :] - mean_decoded)**2)}')
    sample_and_eval(mean_traj, var_traj, vae, y_true, latent_dim, start_len)


def get_approx(latent_model, latent_dim, task, dt):
    """Define the approximation model.

    Inputs
    ------
        latent_model: SDEModel,
            Model to be used for computing
            drift and diffusion for the approximation.
        latent_dim: int,
            Number of latent dimensions
        task: str,
            Name of task, assumed to include a description
            of the approximation to be used. Currently
            supported options include {'lin_approx',
            'sigma_point_3rdcubature', 'sigma_point_unscented'}

    Outputs
    ------
        approx_model: SDEApprox,
            An SDE approximation class.

    """
    if 'lin_approx' in task:
        approx_model = LinearizedApproximationGeneral(latent_model,
                                                      latent_dim=latent_dim)
    elif 'sigma_point_3rdcubature' in task:
        ksi = get_cubature_sigma_points(latent_dim)
        w = get_cubature_weights(latent_dim)
        approx_model = SigmaPointApprox(weights=w,
                                        sigma_pnts=ksi,
                                        model=latent_model,
                                        latent_dim=latent_dim)
    elif 'sigma_point_unscented' in task:
        ksi = get_unscented_sigma_points(latent_dim)
        w = get_unscented_weights(latent_dim)
        approx_model = SigmaPointApprox(weights=w,
                                        sigma_pnts=ksi,
                                        model=latent_model,
                                        latent_dim=latent_dim)
    else:
        raise NotImplementedError(f'Task {task} does not have a matching SDEApprox class.')
    return approx_model


def train_latent_model(
        latent_model, data, val_data, dt, task, latent_dim, context_dim, vae, lr, epochs=100,
        start_len=3,  prior_model=None, decoder_dist=False):
    """Train latent model and VAE.

    Inputs
    ------
        latent_model: SDEModel,
            A model class for drift and diffusion
        data: tf.Tensor,
            Training data sequences to be used.
        val_data: tf.Tensor,
            Validation data to monitor during training
            to avoid overfitting.
        dt: float,
            Time step length between the observations.
        task: str,
            Name of task, defines the type of approximation used.
        latent_dim: int,
            Number of latent state dimensions.
        context_dim: int,
            Number of latent context dimensions,
        epochs: int,
            Number of training epochs,
        vae: VAETF,
            A VAE class instance to be trained.
        lr: float,
            Learning rate for the optimizer.
        start_len: int,
            Length of starting sequence to encode for initial
            position.
        prior_model: SDEModel,
            Optional prior SDEModel in case it is not included
            in the latent_model class.
        decoder_dist: bool,
            Boolean for if the latent samples are decoded to distributions
            instead of points.

    Outputs
    ------
        latent_model: SDEModel,
            A trained SDEModel instance.
    """
    sys.argv = [sys.argv[0]]   # ugly fix to make sure that tf does not get confused by flags.
    approx_dim = latent_dim + context_dim
    approx_model = get_approx(latent_model, approx_dim, task, dt)
    if prior_model is not None:
        approx_prior = get_approx(prior_model, approx_dim, task, dt)
    else:
        approx_prior = None
    approxSDE = ApplyApproxSDE(approx_model,
                            latent_dim=latent_dim,
                            context_dim=context_dim,
                            vae=vae,
                            lr=lr,
                            start_len=start_len,
                            approx_prior=approx_prior,
                            decoder_dist=decoder_dist)
    with  tf.device('/gpu:0'):
        latent_model = approxSDE.train_full(data,
                                            val_data,
                                            epochs=epochs,
                                            dt=dt)

    return latent_model


def test_latent_model(latent_model, vae, data, dt,
                      task, latent_dim, context_dim, start_len=3):
    """Create trajectories from trained latent model.

    Inputs
    ------
        latent_model: SDEModel,
            A trained model for posterior drift and diffusion
        vae: VAETF,
            A trained VAE for encoding the input data
        data: tf.Tensor,
            Input data for generating trajectories. Only the first states
            are used to get an initial point for the mean and variance.
        dt: float,
            The time step size over which the ODE trajectories are
            evaluated.
        task: str,
            Name of the task, defines the approximation class used.
        latent_dim: int,
            Number of latent state dimensions.
        context_dim: int,
            Number of latent context dimensions.
        start_len: int,
            Length of starting sequence to encode for initial
            position.

    Outputs
    ------
        means: tf.Tensor,
            An ODE mean trajectory based on given initial point.
        vars: tf.Tensor,
            An ODE variance trajectory based on given initial point.
    """
    approx_dim = latent_dim + context_dim
    approx_model =  get_approx(latent_model, approx_dim,  task, dt)
    approxSDE = ApplyApproxSDE(approx_model,
                            latent_dim=latent_dim,
                            vae=vae,
                            context_dim=context_dim,
                            start_len=start_len)
    means, covs = approxSDE.generate(data, latent_model, dt)
    return means, covs


def latent_to_obs(latents, vae):
    """Map latent trajectory to observation space.

    Inputs
    -------
        latents: tf.Tensor,
            Latent trajectory (possibly batched),
            to be decoded. Assumed to be 3-D
            (batch_size, sequence_length, dim)
        vae: VAETF,
            Trained VAE class to use for decoding.

    Outputs
    -------
        traj_decoded: tf.Tensor,
            Decoded trajectories.
    """
    traj_decoded = []
    for i in range(latents.shape[1]):
        latent = latents[:, i]
        pred_obs = vae.predict_decoder(latent)
        traj_decoded.append(pred_obs)
    traj_decoded =  tf.stack(traj_decoded, axis=1)
    traj_decoded = tf.cast(traj_decoded, dtype=tf.float64)
    return traj_decoded


def sample_latent_trajectories(means, covs, latent_dim, n_samples=1):
    """Sample latent trajectories based on mean and variance.

    Inputs
    ------
        means: tf.Tensor,
            ODE mean trajectory from an
            approximation
        covs: tf.Tensor,
            ODE variance trajectory from an
            approximation.
        latent_dim: int,
            Number of dimensions
        n_samples: int,
            Number of samples per point to generate.

    Outputs
    -------
        traj: tf.Tensor,
            A tensor of shape (n_samples, batch_size, sequence_length, n_dim),
            the samples from the weak distribution.
    """
    batch_size = means.shape[0]
    seq_len = means.shape[1]
    traj = []
    for i in range(batch_size):
        batch_list = []
        for j in range(seq_len):
            mean = means[i, j]
            cov = tf.reshape(covs[i, j], (latent_dim, latent_dim))
            dist = tfp.distributions.MultivariateNormalTriL(loc=mean,
                                                            scale_tril=tf.linalg.cholesky(cov))
            samples = dist.sample(n_samples)
            batch_list.append(samples)
        batch_samples = tf.stack(batch_list, axis=1)
        traj.append(batch_samples)
    traj = tf.stack(traj, axis=1)
    return traj


def get_sampled_pred_obs(traj, vae):
    """Map latent samples to observation space.

    Inputs
    ------
        traj: tf.Tensor,
            Samples in the latent space, 4 dimensions,
            (n_samples, batch_size, sequence_length, n_dim).
        vae: VAETF,
            A Trained VAE to use in decoding.

    Outputs
    -------
        decoded_samples: tf.Tensor,
            Decoded samples in the observation
            space.

    """
    n_traj = traj.shape[0]
    decoded_samples = []
    for i in range(n_traj):
        pred_decoded = latent_to_obs(traj[i], vae)
        decoded_samples.append(pred_decoded)
    decoded_samples = tf.stack(decoded_samples, axis=0)
    return decoded_samples

def get_mses(samples, y_true, start_len):
    """Compute MSE between samples and true data.

    Inputs
    ------
        samples: tf.Tensor,
            Observation space samples,
            assumed to have number of samples
            as the first dimension.
        y_true: tf.Tensor,
            True data to be compared to.
        start_len: int,
            Length of starting sequence to encode for initial
            position.

    Outputs
    -------
        mses: tf.Tensor,
            A vector of mses, for each sample.
    """
    mses = []
    for i in range(samples.shape[0]):
        mse = tf.reduce_mean((samples[i] - y_true[:, start_len:])**2)
        mses.append(mse)
    mses = tf.stack(mses, axis=0)
    return mses

def sample_and_eval(means, covs, vae, y_true, latent_dim, start_len=3):
    """Sample from the latent distribution and evaluate MSE.

    Inputs
    ------
        means: tf.Tensor,
            An ODE mean trajectory, generated by
            an approximation scheme.
        covs: tf.Tensor,
            An ODE variance trajectory, generated
            by an approximation scheme.
        vae: VAETF,
            A trained VAE to be used for decoding.
        y_true: tf.Tensor,
            The true observations to be compared to.
        latent_dim: int,
            Number of latent dimensions.
        start_len: int,
            Length of starting sequence to encode for initial
            position.
    """
    latent_samples = sample_latent_trajectories(means, covs, latent_dim, n_samples=50)
    decoded_samples = get_sampled_pred_obs(latent_samples, vae)
    error_mse = get_mses(decoded_samples, y_true, start_len)
    error_mse = error_mse.numpy()
    confidence_level = 0.95
    deg_free = error_mse.shape[0] - 1
    sample_mean = np.mean(error_mse)
    sample_standard_error = scipy.stats.sem(error_mse)
    confidence_interval = scipy.stats.t.interval(confidence_level, deg_free, sample_mean, sample_standard_error)

    print(f'Mean MSE of 50 samples {sample_mean}')
    print(f'95% confidence interval of MSE based on 50 samples {confidence_interval[1] - sample_mean}')


def get_data(base_folder, vae_name):
    """Retrieve data and VAE paths.

    Inputs
    ------
        base_folder: str,
            A folder containing a folder with name 'data/walking',
            where the data is assumed to be located
        vae_name: str,
            Name of the vae model, used in the encoder&decoder path
            definitions.

    Outputs
    ------
        dataset_train: CMUWalkingTF,
            Dataset object for training
        dataset_val: CMUWalkingTF,
            Dataset object for validation
        dataset_test: CMUWalkingTF,
            Dataset object for testing
        enc_path: str,
            Path to VAE encoder model checkpoints
        dec_path: str,
            Path to VAE decoder model checkpoints
    """
    data_path = os.path.join(base_folder, 'data', 'walking')
    enc_name = f'encoder_{vae_name}'
    dec_name = f'decoder_{vae_name}'
    dataset_train = CMUWalkingTF(data_path, data_section='train')
    dataset_val = CMUWalkingTF(data_path, data_section='val')
    dataset_test = CMUWalkingTF(data_path, data_section='test')


    enc_path = os.path.join(base_folder, 'model', enc_name)
    dec_path = os.path.join(base_folder, 'model', dec_name)

    return dataset_train, dataset_val, dataset_test, enc_path, dec_path

