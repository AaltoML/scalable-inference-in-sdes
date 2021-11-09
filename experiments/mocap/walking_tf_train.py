import os
import sys
sys.path.append('')
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('-base_folder', type=str, help='Folder containing a data folder')
parser.add_argument('-task', type=str, default='lin_approx', help='Name of approximation type, see walking_tf_functions.py get_approx for supported')
parser.add_argument('-decoder_dist', type=str, default='False', help='If the decoder outputs a distribution or a point.')
parser.add_argument('-model_name', type=str, help='Model name, to be used as a saved model file name')
parser.add_argument('-vae_name', type=str, help='VAE name, to be used as a saved VAE file name')

parser.add_argument('-prior_model_name', type=str,  help='Prior model name, to be used as saved prior file name. If includes str "fit", the prior dirft is fit.')
parser.add_argument('--lr', type=float, default=0.01,  help='Learning rate for the model.')
parser.add_argument('--dt', type=float, default=0.1, help='Assumed time difference between observations')
parser.add_argument('--latent_dim', type=int, default=6, help='Number of latent state dimensions')
parser.add_argument('--context_dim', type=int, default=3, help='Number of latent context dimensions')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to use in training')
parser.add_argument('--start_len', type=int, default=3, help='Number of data points in beginning of sequence to encode.')
args = parser.parse_args()

sys.path.append(args.base_folder)
from walking_tf_functions import get_vae, get_latent_model, get_data, train_latent_model

decoder_dist = args.decoder_dist == 'True'

# define loaders and datasets
output_path = os.path.join(args.base_folder, "model_data", args.model_name)
dataset_train, dataset_val, dataset_test, enc_path, dec_path,  = get_data(args.base_folder, args.vae_name)
eval_times = dataset_train.x
obs_dim = dataset_train.y.shape[-1] # assuming flat data

# get vae
vae = get_vae(args.latent_dim, obs_dim,  args.context_dim, encoder_path=enc_path, decoder_path=dec_path,
              decoder_dist=decoder_dist)


# get latent models (posterior and prior)
latent_model = get_latent_model( output_path, args.latent_dim, args.context_dim, prior=False)
prior_full_path = os.path.join(args.base_folder, "model_data", args.prior_model_name)
print(f'Model path: {output_path}')
print(f'Prior model path: {prior_full_path}')
prior_model = get_latent_model(prior_full_path, args.latent_dim, args.context_dim, prior=True)

# run test
end_time = args.dt*dataset_train.y.shape[1]
latent_model = train_latent_model(latent_model,  dataset_train.y, dataset_val.y, args.dt,
                                  args.task, args.latent_dim, args.context_dim, vae,
                                  epochs=args.epochs, start_len=args.start_len,
                                  lr=args.lr,
                                  prior_model=prior_model,
                                  decoder_dist=decoder_dist)
