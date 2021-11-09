import os
from .walking_tf_functions import get_vae, get_latent_model, get_data, test_latent_model, eval_test_performance
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('-base_folder', type=str, help='Folder containing a data folder')
parser.add_argument('-task', type=str, default='lin_approx', help='Name of approximation type, see walking_tf_functions.py get_approx for supported')
parser.add_argument('-decoder_dist', type=str, default='False', help='If the decoder outputs a distribution or a point.')
parser.add_argument('-model_name', type=str, help='Model name, to be used as a saved model filename')
parser.add_argument('-vae_name', type=str, help='VAE name, to be used in saved VAE file name')
parser.add_argument('--dt', type=float, default=0.1, help='Assumed time difference between observations')
parser.add_argument('--latent_dim', type=int, default=6, help='Number of latent state dimensions')
parser.add_argument('--context_dim', type=int, default=3, help='Number of latent context dimensions')
parser.add_argument('--start_len', type=int, default=3, help='Number of data points in beginning of sequence to encode.')
args = parser.parse_args()

decoder_dist = args.decoder_dist == 'True'

# define loaders and datasets
output_path = os.path.join(args.base_folder,  "model_data", args.model_name)
dataset_train, dataset_val, dataset_test,  enc_path, dec_path,  = get_data(args.base_folder, args.vae_name)
obs_dim = dataset_train.y.shape[-1] # assuming flat data

# get vae
vae = get_vae(args.latent_dim, obs_dim,  args.context_dim, encoder_path=enc_path, decoder_path=dec_path,
              decoder_dist=decoder_dist)


# get latent model
latent_model = get_latent_model( output_path, args.latent_dim, args.context_dim, prior=False)

# run test
mean_samples, var_samples = test_latent_model(latent_model, vae, dataset_test.y, args.dt,
                                               args.task, args.latent_dim,
                                              args.context_dim, start_len=args.start_len)

# evaluate and plot
eval_test_performance(
    args.base_folder, args.task, mean_samples, var_samples,  vae, dataset_test.y, args.latent_dim, start_len=args.start_len)