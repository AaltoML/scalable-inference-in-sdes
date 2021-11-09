import os

import torchvision
import torchvision.transforms as transforms
import torch
import numpy as np
from scipy.ndimage import rotate
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.stats import multivariate_normal

np.random.seed(43)


def rotate_img(img, angles: list):
    """ Rotate the input MNIST image in angles specified """
    rotated_imgs = np.array(img).reshape((-1, 1, 28, 28))
    for a in angles:
        rotated_imgs = np.concatenate(
            (
                rotated_imgs,
                rotate(img, a, axes=(1, 2), reshape=False).reshape((-1, 1, 28, 28)),
            ),
            axis=1,
        )
    return rotated_imgs


def create_rotating_dataset(data_path, digit=3, train_n=100, test_n=10, n_angles=64):
    """
        Takes the MNIST data path as input and returns the rotating data by rotating the digit uniformly
        for n_angles angles.
    """
    mnist_train = torchvision.datasets.mnist.MNIST(
        data_path, download=True, transform=transforms.ToTensor()
    )
    mnist_test = torchvision.datasets.mnist.MNIST(
        data_path, download=True, train=False, transform=transforms.ToTensor()
    )

    angles = np.linspace(0, 2 * np.pi, n_angles)[1:]
    angles = np.rad2deg(angles)

    train_digit_idx = torch.where(mnist_train.train_labels == digit)
    train_digit_imgs = mnist_train.train_data[train_digit_idx]
    random_idx = np.random.randint(0, train_digit_imgs.shape[0], train_n)
    train_digit_imgs = train_digit_imgs[random_idx]
    train_rotated_imgs = rotate_img(train_digit_imgs, angles)
    train_rotated_imgs = train_rotated_imgs / 255
    train_rotated_imgs = train_rotated_imgs.astype(np.float32)

    test_digit_idx = torch.where(mnist_test.test_labels == digit)
    test_digit_imgs = mnist_test.train_data[test_digit_idx]
    random_idx = np.random.randint(0, test_digit_imgs.shape[0], test_n)
    test_digit_imgs = test_digit_imgs[random_idx]
    test_rotated_imgs = rotate_img(test_digit_imgs, angles)
    test_rotated_imgs = test_rotated_imgs / 255

    test_rotated_imgs = test_rotated_imgs.astype(np.float32)

    return train_rotated_imgs, test_rotated_imgs


def get_device():
    """
        Returns the available device required for torch.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def kl_divergence(mean, logvar):
    """
    KL Divergence value between the input distribution specified with mean and logvar and N(0,I) considering
    diagonal covariance.
    """
    var = torch.exp(logvar)
    mean2 = mean * mean
    loss = -0.5 * torch.mean(1 + logvar - mean2 - var)
    return loss


def visualize_embeddings(encoder, dataloader, n_samples, device, n_classes=16, output_path=None):
    """Visualize the embeddings in the latent space"""
    # classes = list(np.linspace(0, n_classes-1, n_classes).astype(np.str))
    n = 0
    codes, labels = [], []
    with torch.no_grad():
        for b_inputs, b_labels in dataloader:
            batch_size = b_inputs.size(0)
            b_codes = encoder(b_inputs.to(device))[0]
            b_codes, b_labels = b_codes.cpu().data.numpy(), b_labels.cpu().data.numpy()
            if n + batch_size > n_samples:
                codes.append(b_codes[: n_samples - n])
                labels.append(b_labels[: n_samples - n])
                break
            else:
                codes.append(b_codes)
                labels.append(b_labels)
                n += batch_size
    codes = np.vstack(codes)
    if codes.shape[1] > 2:
        codes = TSNE().fit_transform(codes)
    labels = np.hstack(labels)

    fig, ax = plt.subplots(1)
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 0.8, pos.height], which="both")
    color_map = plt.cm.get_cmap('hsv', n_classes)

    for iclass in range(min(labels), max(labels) + 1):
        ix = labels == iclass
        ax.plot(codes[ix, 0], codes[ix, 1], ".", c=color_map(iclass))

    # plt.legend(classes, bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.tight_layout()
    plt.suptitle("Latent embeddings of a sample datapoint using HSV colorcode", y=1)
    if output_path is None:
        plt.show()
    else:
        plt.savefig(os.path.join(output_path, "latent-embeddings.png"))


def visualize_output(vae, x, output_path=None):
    y = vae.test(x)

    y = y.cpu().detach().numpy().reshape(16, 28, 28)

    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    for ax, img in zip(axs.flat, x.cpu()):
        ax.imshow(img.reshape(28, 28), cmap="gray")
        ax.axis('off')

    plt.suptitle("Original image", y=1)
    plt.tight_layout()
    if output_path is None:
        plt.show()
    else:
        plt.savefig(os.path.join(output_path, "vae-original.png"))

    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    for ax, img in zip(axs.flat, y):
        ax.imshow(img.reshape(28, 28), cmap="gray")
        ax.axis('off')
    plt.suptitle("Predicted image", y=1)
    plt.tight_layout()

    if output_path is None:
        plt.show()
    else:
        plt.savefig(os.path.join(output_path, "vae-prediction.png"))


def create_gp_dataset(vae, x, n=-1, n_classes=16, img_shape=(28, 28),
                      device=torch.device("cpu")):
    """Create the dataset for GP by encoding the data via VAE encoder"""
    x = torch.tensor(x, dtype=torch.float32, device=device)

    # More than 1 batch size doesn't work
    dataset_loader = torch.utils.data.DataLoader(x, batch_size=1, shuffle=False)

    x_gp = None
    y_gp = None
    true_x = None

    for x in dataset_loader:
        x_gp_temp = vae.predict_encoder(x.view(-1, 1, *img_shape))

        y_gp_temp = torch.cat((x_gp_temp[1:], x_gp_temp[0].view(1, -1))).view(
            (1, n_classes, -1)
        )
        y_gp = y_gp_temp if y_gp is None else torch.cat((y_gp, y_gp_temp))

        x_gp = (
            x_gp_temp.view(1, n_classes, -1)
            if x_gp is None
            else torch.cat((x_gp, x_gp_temp.view(1, n_classes, -1)))
        )

        true_x = (
            x.view(-1, n_classes, *img_shape)
            if true_x is None
            else torch.cat((true_x, x.view(-1, n_classes, *img_shape)))
        )

    if n == -1:
        return (
            x_gp.detach().cpu().numpy().astype("float64"),
            y_gp.detach().cpu().numpy().astype("float64"),
            true_x,
        )
    else:
        assert x_gp.shape[0] > n
        random_idx = np.random.randint(0, x_gp.shape[0], n)
        x_gp = x_gp[random_idx]
        y_gp = y_gp[random_idx]
        true_x = true_x[random_idx]
        return (
            x_gp.detach().numpy().astype("float64"),
            y_gp.detach().numpy().astype("float64"),
            true_x,
        )


def perform_gp_euler(gp_model, x, dt, tn, n_trajectories=1):
    """
    Perform GP inference using Euler Scheme.
    Note: This function only returns the values at integer time-points like t=0,1,2...
    """
    gp_predictions = None
    n_steps = int(tn / dt) + 1
    last_gp_pred = np.repeat(x.reshape((1, x.shape[-1])), n_trajectories, axis=0)
    for i in range(1, n_steps):
        gp_mean, gp_S = gp_model.predict_mean_var(last_gp_pred)

        gp_S = gp_S.reshape(-1)
        gp_S = np.diag(gp_S)
        l = np.sqrt(gp_S)  # As S is a diagonal matrix for independent GP

        db = np.random.normal(size=gp_S.shape[0]).reshape((-1, 1))
        gp_pred = last_gp_pred + (dt * gp_mean) + np.sqrt(dt) * (l @ db).reshape((-1, gp_mean.shape[-1]))

        last_gp_pred = gp_pred
        if (i * dt).is_integer():
            if gp_predictions is None:
                gp_predictions = last_gp_pred.reshape((-1, 1, last_gp_pred.shape[-1]))
            else:
                gp_predictions = np.concatenate([gp_predictions, last_gp_pred.reshape((-1, 1, last_gp_pred.shape[-1]))],
                                                axis=1)
            print(f"t = {i * dt}", end='\r')

    print(f"Euler-Maruyama inference completed...")

    return gp_predictions


def generate_velocity_vector_data(X, Y, normalize=False):
    """Function to create the GP velocity vector data for training the GP(vector field)"""
    y_tr_vel = []
    for i in range(X.shape[0]):
        x = X[i, :]
        y = Y[i, :]
        direction = (y - x).reshape((1, -1))
        if normalize:
            unit_vector = direction / np.linalg.norm(direction)
            direction = 1 * unit_vector  # x + 1 * unit_vector
        y_tr_vel.append(direction)
    Y = np.array(y_tr_vel).reshape((-1, Y.shape[-1]))
    return Y


def mse_loss(y1, y2):
    """Calculates the MSE loss between y1 and y2"""
    loss = torch.nn.MSELoss()
    y1 = torch.tensor(y1)
    y2 = torch.tensor(y2)
    mse_val = loss(y1, y2)
    return mse_val.numpy()


def calculate_gaussian_nlpd(val, m, S):
    """Calculate Gaussian NLPD for diagonal covariance"""
    val = val.reshape(-1)
    m = m.reshape(-1)
    assert m.shape == val.shape
    S = S.reshape((m.shape[0], m.shape[0]))
    S = np.diag(S)

    return -multivariate_normal(m, S).logpdf(val)
