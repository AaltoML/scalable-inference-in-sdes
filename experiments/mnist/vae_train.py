""" Script to train the VAE model """
import os

import torch
import numpy as np

from vae_model import VAE
from utility import get_device


class Dataset(torch.utils.data.Dataset):
    """Dataset class for torch, used to make dataloader"""
    def __init__(self, x, y):
        self.x = x.astype(np.float32)
        self.y = y.reshape(-1)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        x_idx = self.x[index]
        y_idx = self.y[index]

        return x_idx, y_idx


def load_rotating_mnist_data(data_path, n_angels, batch_size=64):
    """
        Load the rotating MNIST data from the given path and return the torch train and test dataloader.
        Label for the data is the angle index.
    """
    x_true = np.load(data_path).reshape((-1, 1, 28, 28))

    t = np.linspace(0, n_angels - 1, n_angels).astype(np.uint8).reshape((1, -1))

    tr_t = np.repeat(t, x_true.shape[0] // n_angels, axis=0).reshape((-1, 1))
    tr_dataset = Dataset(x_true, tr_t)

    data_loader = torch.utils.data.DataLoader(
        tr_dataset, batch_size=batch_size, shuffle=True
    )

    return data_loader


def vae_train(rotating_mnist_train_dataset, epochs, output_model_path, latent_dim=16, bce_weighted_loss=1000,
              lr=0.001, n_angles=64):

    device = get_device()
    # Load data
    train_loader = load_rotating_mnist_data(rotating_mnist_train_dataset,
                                                         n_angels=n_angles)

    if not os.path.exists(output_model_path):
        os.makedirs(output_model_path)

    vae = VAE(device, latent_dim=latent_dim)

    encoder_path = os.path.join(output_model_path, "encoder.pt")
    decoder_path = os.path.join(output_model_path, "decoder.pt")

    if os.path.exists(encoder_path) and os.path.exists(decoder_path):
        print("Loading weights...")
        vae.load(encoder_path, decoder_path)

    # print model summary
    vae.print_summary()

    if epochs > 0:
        vae.train(
            train_loader,
            epochs=epochs,
            device=device,
            lr=lr,
            bce_weighted_loss=bce_weighted_loss,
        )
        vae.save(encoder_path, decoder_path)

    return vae
