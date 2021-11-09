from functools import reduce

import torch.nn as nn
import torch
import numpy as np
from torchsummary import summary

from utility import kl_divergence


class Encoder(nn.Module):
    def __init__(self, latent_dim=16, img_shape=(28, 28)):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        in_features = 8 * reduce(lambda x, y: (x - 4) * (y - 4), img_shape)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=in_features, out_features=latent_dim)
        self.fc2 = nn.Linear(in_features=in_features, out_features=latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)

        mean = self.fc1(x)
        log_var = self.fc2(x)
        return mean, log_var

    def draw_sample(self, z_mean, z_logvar):
        samples = torch.randn_like(z_mean)
        samples = z_mean + samples * (torch.exp(0.5 * z_logvar))
        return samples


class Decoder(nn.Module):
    def __init__(self, latent_dim=16, img_shape=(28, 28)):
        super(Decoder, self).__init__()
        out_features = 8 * reduce(lambda x, y: (x - 4) * (y - 4), img_shape)
        self.conv_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=3, stride=1),
        )

        self.img_shape = img_shape
        self.fc1 = nn.Linear(in_features=latent_dim, out_features=out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        img_shapes = [x - 4 for x in self.img_shape]
        x = x.reshape(x.shape[0], 8, *img_shapes)
        x = self.conv_layer(x)

        return torch.sigmoid(x)


class VAE:
    def __init__(self, device="cpu", latent_dim=16, img_shape=(28, 28), encoder_shape=None,
                 decoder_shape=None, encoder_path=None,
                 decoder_path=None):
        if encoder_shape is None:
            encoder_shape = img_shape
        if decoder_shape is None:
            decoder_shape = img_shape

        self.encoder_path = encoder_path
        self.decoder_path = decoder_path

        self.encoder = Encoder(latent_dim, img_shape=encoder_shape).to(device)
        self.decoder = Decoder(latent_dim, img_shape=decoder_shape).to(device)
        self.latent_dim = latent_dim
        self.img_shape = img_shape

        self.loss = nn.BCELoss(reduction="mean")

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def nll(self, x, mean, log_var):
        # term1 = (-(28*28)/2) * torch.log(2*np.pi)
        term2 = -(1 / 2) * torch.sum(log_var)
        term3 = -(1 / 2) * torch.sum((x - mean) ** 2 / (torch.exp(log_var)))

        return -(term2 + term3) / x.shape[0]

    def train(self, train_loader, epochs, lr=0.001, device=None, bce_weighted_loss=1000):
        """Train the VAE model."""
        optim = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=lr
        )

        print("------------------------------------------------------------------------------------")
        print("VAE Train")
        print("------------------------------------------------------------------------------------")
        for i in range(epochs):
            running_loss = []
            self.encoder.train()
            self.decoder.train()
            for (x, _) in train_loader:
                optim.zero_grad()
                x = x.to(device)
                enc_mean, enc_logvar = self.encoder(x)
                z = self.encoder.draw_sample(enc_mean, enc_logvar)
                kl = kl_divergence(enc_mean, enc_logvar)

                y = self.decoder(z)
                bce_loss_val = bce_weighted_loss * self.loss(y, x)
                loss_val = kl + bce_loss_val

                loss_val.backward()
                optim.step()
                running_loss.append(loss_val.item())

            print(f"Training Loss : Epoch {i + 1} : {np.mean(running_loss)}",  end='\r')

        if epochs > 0:
            print(f"Training Loss : Epoch {epochs} : {np.mean(running_loss)}")
        print("------------------------------------------------------------------------------------")

    def test(self, x):
        """Test the VAE model on data x. First x is encoded using encoder model, a sample is produced from then latent
        distribution and then it is passed through the decoder model."""
        self.encoder.eval()
        self.decoder.eval()
        enc_m, enc_log_var = self.encoder(x)
        z = self.encoder.draw_sample(enc_m, enc_log_var)
        y = self.decoder(z)
        return y

    def load(self, encoder_path=None, decoder_path=None):
        """Load the weights of encoder and decoder"""
        if encoder_path is None:
            encoder_path = self.encoder_path
        if decoder_path is None:
            decoder_path = self.decoder_path
        self.encoder.load_state_dict(torch.load(encoder_path))
        self.decoder.load_state_dict(torch.load(decoder_path))

    def save(self, encoder_path=None, decoder_path=None):
        """Save the VAE model. Both encoder and decoder and saved in different files."""
        if encoder_path is None:
            encoder_path = self.encoder_path
        if decoder_path is None:
            decoder_path = self.decoder_path
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.decoder.state_dict(), decoder_path)

    def print_summary(self):
        """Print the summary of both the models: encoder and decoder"""
        summary(self.encoder, (1, *self.img_shape))
        summary(self.decoder, (1, self.latent_dim))

    def predict_encoder(self, x):
        self.encoder.eval()

        z_s0_mean, z_s0_logvar = self.encoder(x)
        z_s0 = self.encoder.draw_sample(z_s0_mean, z_s0_logvar)

        return z_s0

    def predict_decoder(self, z):
        self.decoder.eval()
        return self.decoder(z)

    def encoder_jacobian(self, x):
        jacobian_enc = torch.autograd.functional.jacobian(self.encoder.forward, x)
        return jacobian_enc

    def decoder_jacobian(self, x):
        jacobian_dec = torch.autograd.functional.jacobian(self.decoder.forward, x)
        return jacobian_dec
