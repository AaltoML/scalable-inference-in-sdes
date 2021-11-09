from gpflow.utilities import print_summary

import sys
sys.path.append("../../")
from src.gp.model import LatentSVGP


def train_svgp(x_tr, y_tr, output_path, epochs, lr=0.01, latent_dim=16,
               svgp_inducing_pnts=1000, k_lengthscale=1.0, k_variance=1.):
    latent_gp = LatentSVGP(
        x_tr,
        output_path=output_path,
        lr=lr,
        num_latent_gps=latent_dim,
        n_inducing_pnts=svgp_inducing_pnts,
        k_lengthscale=k_lengthscale,
        k_variance=k_variance,
    )
    print_summary(latent_gp.model)
    latent_gp.train(x_tr, y_tr, epochs=epochs, minibatch_size=256)

    return latent_gp
