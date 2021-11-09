import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D

from gp_train import train_svgp
from vae_train import vae_train, load_rotating_mnist_data
from utility import get_device, visualize_output, visualize_embeddings, create_rotating_dataset, create_gp_dataset, generate_velocity_vector_data, perform_gp_euler, mse_loss, calculate_gaussian_nlpd
from linearized_approximation_mnist import LinearizedApproximationMNIST
from sigma_point_approx_mnist import SigmaPointApproxMNIST
import config

import sys
sys.path.append("../..")
from src.sde_tf.sde_approx.get_sigma_points import get_cubature_sigma_points, get_cubature_weights

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # or any {'0', '1', '2'}

np.random.seed(43)

if __name__ == '__main__':

    vae_epochs = config.vae_epochs
    n_angle = config.n_angle
    gp_epochs = config.gp_epochs
    output_path = config.output_path
    latent_dim = config.latent_dim
    dt = config.dt
    end_t = config.end_t
    euler_trajectories = config.euler_trajectories

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    device = get_device()

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create the rotating dataset
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    rotating_mnist_train_dataset = os.path.join(output_path, "rotating_mnist_train_3_64_angles.npy")
    rotating_mnist_test_dataset = os.path.join(output_path, "rotating_mnist_test_3_64_angles.npy")

    if os.path.exists(rotating_mnist_train_dataset) and os.path.exists(rotating_mnist_test_dataset):
        train_rotated_imgs = np.load(rotating_mnist_train_dataset)
        test_rotated_imgs = np.load(rotating_mnist_test_dataset)
    else:
        train_rotated_imgs, test_rotated_imgs = create_rotating_dataset(output_path, digit=3, train_n=config.n_train,
                                                                        test_n=config.n_test, n_angles=n_angle)
        np.save(rotating_mnist_train_dataset, train_rotated_imgs)
        np.save(rotating_mnist_test_dataset, test_rotated_imgs)

    # visualize
    sample_img = train_rotated_imgs[1].reshape(-1, 28, 28)
    _, axs = plt.subplots(1, n_angle, figsize=(120, 5))
    for i in range(n_angle):
        axs[i].imshow(sample_img[i].reshape((28, 28)), cmap="gray")
        axs[i].axis('off')
        axs[i].set_title(f"t={i}")
    plt.suptitle("Dataset Sample", fontsize=24)
    plt.tight_layout(h_pad=0, w_pad=0)
    plt.savefig(os.path.join(output_path, "sample-dataset.png"))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Train VAE
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    vae_model_path = os.path.join(output_path, "MNIST-VAE")
    vae = vae_train(rotating_mnist_train_dataset, epochs=vae_epochs, output_model_path=vae_model_path,
                    latent_dim=latent_dim)

    test_loader = load_rotating_mnist_data(rotating_mnist_test_dataset, n_angels=n_angle)

    x = iter(test_loader).next()[0].to(device)
    # only first 16 images
    visualize_output(
        vae, x[:16], output_path
    )

    visualize_embeddings(
        vae.get_encoder(), test_loader, 1000, device, n_classes=n_angle, output_path=output_path
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create GP dataset
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    x_tr_true = np.load(rotating_mnist_train_dataset)
    x_te_true = np.load(rotating_mnist_test_dataset)

    x_tr, y_tr, _ = create_gp_dataset(
        vae,
        x_tr_true,
        n_classes=n_angle,
        device=device
    )
    x_te, _, _ = create_gp_dataset(
        vae,
        x_te_true,
        n_classes=n_angle,
        device=device
    )

    # Reshape the datasets
    x_tr = x_tr.reshape((-1, x_tr.shape[-1]))
    y_tr = y_tr.reshape((-1, y_tr.shape[-1]))
    x_te = x_te.reshape((-1, x_te.shape[-1]))

    # GP velocity vector dataset
    y_tr = generate_velocity_vector_data(x_tr, y_tr)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Train GP
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    gp_model_path = os.path.join(output_path, "MNIST-GP")
    latent_gp = train_svgp(x_tr, y_tr, gp_model_path, epochs=gp_epochs)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Inference
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    test_random_idx = np.random.randint(0, x_te.shape[0], 1)
    test_random_idx = (
            test_random_idx - test_random_idx % n_angle
    )  # To always get the first image of the sequence
    test_random_idx = test_random_idx[0]

    z0 = x_te[test_random_idx].reshape((-1, latent_dim))

    total_preds = int(end_t)
    ground_truth_images = x_te_true.reshape((-1, 28, 28))[test_random_idx:test_random_idx + total_preds, :, :]

    # Setting t=64 as t=0 as done at the time of inference
    ground_truth_images = np.concatenate((ground_truth_images[1:].reshape((-1, 1, 28, 28)),
                                          ground_truth_images[0].reshape((1, 1, 28, 28))), axis=0)

    z_last = x_te[test_random_idx + total_preds - 1].reshape((-1, latent_dim))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Euler-Maruyama
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("------------------------------------------------------------------------------------")
    print("Performing Euler-Maruyama...")
    print("------------------------------------------------------------------------------------")
    gp_input = z0
    pred_gp_euler = perform_gp_euler(latent_gp, gp_input, dt, end_t, n_trajectories=euler_trajectories)

    np.savez(os.path.join(output_path, "em-predictions"), predictions=pred_gp_euler)
    pred_gp_euler_mean = np.mean(pred_gp_euler, axis=0).reshape((-1, latent_dim))
    print("------------------------------------------------------------------------------------")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Linearized Inference
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("------------------------------------------------------------------------------------")
    print("Performing Linearized Approximation...")
    print("------------------------------------------------------------------------------------")
    linearized_approx = LinearizedApproximationMNIST(latent_gp, latent_dim)

    m0_arr = z0.reshape(-1)
    P0_arr = (np.eye(latent_dim) * 10e-6).reshape(-1)
    ode_linearized_euler_values = []

    last_val = np.concatenate((m0_arr, P0_arr))
    for state in range(1, int(end_t / dt) + 1):
        current_val = linearized_approx.get_nxt_data(
            t=tf.convert_to_tensor(state), x=tf.convert_to_tensor(last_val.reshape((1, -1)))
        )
        current_val = current_val.numpy()
        current_val = current_val.reshape((-1))
        last_val = last_val + dt * current_val

        if (state * dt).is_integer():
            ode_linearized_euler_values.append(last_val)
            print(f"t = {state * dt}", end='\r')

    print(f"Linearization approximation inference completed...")
    ode_linearized_euler_m = np.array(ode_linearized_euler_values)[:, :latent_dim]
    ode_linearized_euler_P = np.array(ode_linearized_euler_values)[:, latent_dim:]

    np.savez(os.path.join(output_path, "lin-predictions"), mean_predictions=ode_linearized_euler_m,
             cov_predictions=ode_linearized_euler_P)
    print("------------------------------------------------------------------------------------")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Sigma Point Approximation
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("------------------------------------------------------------------------------------")
    print("\nPerforming Sigma Point Approximation...")
    print("------------------------------------------------------------------------------------")
    weights = get_cubature_weights(latent_dim)
    sigma_pnts = get_cubature_sigma_points(latent_dim).numpy()
    sigma_approx = SigmaPointApproxMNIST(weights=tf.convert_to_tensor(weights),
                                         sigma_pnts=tf.convert_to_tensor(sigma_pnts), model=latent_gp)

    m0_arr = z0.reshape(-1)
    P0_arr = (np.eye(latent_dim) * 10e-6).reshape(-1)
    ode_sigma_euler_values = []
    last_val = np.concatenate((m0_arr, P0_arr))
    for state in range(1, int(end_t / dt) + 1):
        current_val = sigma_approx.get_nxt_data_one(
            t=tf.convert_to_tensor(state), data=tf.convert_to_tensor(last_val)
        )
        last_val = last_val + dt * current_val
        if (state * dt).is_integer():
            ode_sigma_euler_values.append(last_val)
            print(f"t = {state * dt}", end='\r')

    print(f"Sigma point approximation inference completed...")
    ode_sigma_euler_m = np.array(ode_sigma_euler_values)[:, :latent_dim]
    ode_sigma_euler_P = np.array(ode_sigma_euler_values)[:, latent_dim:]
    np.savez(os.path.join(output_path, "mm-predictions"), mean_predictions=ode_sigma_euler_m,
             cov_predictions=ode_sigma_euler_P)

    print("------------------------------------------------------------------------------------")
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plotting
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("\nSaving plots....")
    em_pred_images = []
    mm_pred_images = []
    linearized_pred_images = []
    em_lst_cov = -1
    _, axs = plt.subplots(7, total_preds, figsize=(120, 15))
    for i in range(0, total_preds):
        gp_pred_mean = pred_gp_euler_mean[i].reshape((1, latent_dim))
        dec_img = (
            vae.predict_decoder(torch.tensor(gp_pred_mean, dtype=torch.float32, device=device)).detach().cpu().numpy()
        ).reshape((28, 28))
        em_pred_images.append(dec_img)

        sde_covar = np.cov(pred_gp_euler[:, i, :].reshape((-1, latent_dim)), rowvar=False)
        jac = vae.decoder_jacobian(torch.tensor(gp_pred_mean, dtype=torch.float32, device=device)).reshape((-1, latent_dim))
        jac = jac.cpu().numpy().astype(np.float64)
        uncertainty = np.sqrt(np.diag(jac @ sde_covar @ jac.T).reshape((28, 28)))
        em_lst_cov = sde_covar

        axs[0][i].imshow(ground_truth_images[i].reshape((28, 28)), cmap="gray")
        axs[0][i].title.set_text(f'Ground Truth t={i+1}')
        axs[0][i].axis('off')

        axs[1][i].imshow(dec_img, cmap="gray")
        axs[1][i].title.set_text('EM Mean')
        axs[1][i].axis('off')

        axs[2][i].imshow(uncertainty, cmap="gray")
        axs[2][i].axis('off')
        axs[2][i].title.set_text('EM std dev')

        val = ode_linearized_euler_m[i].reshape((1, latent_dim))
        dec_img = (
            vae.predict_decoder(torch.tensor(val, dtype=torch.float32, device=device)).detach().cpu().numpy()
        ).reshape((28, 28))
        linearized_pred_images.append(dec_img)

        axs[3][i].imshow(dec_img, cmap="gray")
        axs[3][i].axis('off')
        axs[3][i].title.set_text('Lin. Mean')

        jac = vae.decoder_jacobian(torch.tensor(val, dtype=torch.float32, device=device)).reshape((-1, latent_dim))
        jac = jac.cpu().numpy().astype(np.float64)
        uncertainty = np.sqrt(
            np.diag(jac @ ode_linearized_euler_P[i].reshape((latent_dim, -1)) @ jac.T).reshape((28, 28)))

        axs[4][i].imshow(uncertainty, cmap="gray")
        axs[4][i].axis('off')
        axs[4][i].title.set_text('Lin. std dev')

        val = ode_sigma_euler_m[i].reshape((1, latent_dim))
        dec_img = (
            vae.predict_decoder(torch.tensor(val, dtype=torch.float32, device=device)).detach().cpu().numpy()
        ).reshape((28, 28))
        mm_pred_images.append(dec_img)

        axs[5][i].imshow(dec_img, cmap="gray")
        axs[5][i].axis('off')
        axs[5][i].title.set_text('MM mean')

        jac = vae.decoder_jacobian(torch.tensor(val, dtype=torch.float32, device=device)).reshape((-1, latent_dim))
        jac = jac.cpu().numpy().astype(np.float64)
        uncertainty = np.sqrt(
            np.diag(jac @ ode_sigma_euler_P[i].reshape((latent_dim, -1)) @ jac.T).reshape((28, 28)))

        axs[6][i].imshow(uncertainty, cmap="gray")
        axs[6][i].axis('off')
        axs[6][i].title.set_text('MM std dev')

    plt.suptitle("Outputs of various inference schemes", y=1, fontsize=24)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "inference.png"))

    # Plot trajectory
    fig = plt.figure()
    fig.add_subplot(111, projection="3d")

    true_trajectory = x_te[test_random_idx:test_random_idx + total_preds].reshape((-1, latent_dim))
    euler_trajectory_pnts = pred_gp_euler_mean.reshape((-1, latent_dim))

    std_dev = np.std(true_trajectory, axis=0)
    interesting_dims = np.argsort(std_dev)[-3:][::-1]

    true_traj_dim_3 = true_trajectory[:, interesting_dims]
    euler_pred_traj_dim_3 = euler_trajectory_pnts[:, interesting_dims]
    moment_matching_traj_dim_3 = ode_sigma_euler_m[:, interesting_dims]
    lineaized_traj_dim_3 = ode_linearized_euler_m[:, interesting_dims]

    # Adding the t0 to the predictions
    euler_pred_traj_dim_3 = np.concatenate((true_traj_dim_3[0].reshape((1, -1)), euler_pred_traj_dim_3),
                                           axis=0)
    moment_matching_traj_dim_3 = np.concatenate((true_traj_dim_3[0].reshape((1, -1)), moment_matching_traj_dim_3),
                                                axis=0)
    lineaized_traj_dim_3 = np.concatenate((true_traj_dim_3[0].reshape((1, -1)), lineaized_traj_dim_3),
                                          axis=0)

    plt.plot(true_traj_dim_3[:, 0], true_traj_dim_3[:, 1], true_traj_dim_3[:, 2], label="True trajectory")
    plt.plot(euler_pred_traj_dim_3[:, 0], euler_pred_traj_dim_3[:, 1], euler_pred_traj_dim_3[:, 2], label="EM trajectory")
    plt.plot(moment_matching_traj_dim_3[:, 0], moment_matching_traj_dim_3[:, 1], moment_matching_traj_dim_3[:, 2],
             label="MM trajectory")
    plt.plot(lineaized_traj_dim_3[:, 0], lineaized_traj_dim_3[:, 1], lineaized_traj_dim_3[:, 2],
             label="Lin. trajectory")
    plt.legend()
    plt.suptitle("True and predicted latent trajectory")
    plt.savefig(os.path.join(output_path, "latent-trajectory.png"))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Metric
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ground_truth_images = ground_truth_images.reshape((-1, 1, 28, 28))
    mm_pred_images = np.array(mm_pred_images).reshape(ground_truth_images.shape)
    em_pred_images = np.array(em_pred_images).reshape(ground_truth_images.shape)
    linearized_pred_images = np.array(linearized_pred_images).reshape(ground_truth_images.shape)

    print("------------------------------------------------------------------------------------")
    print("MSE values:")
    print(f"Euler-Maruyama: {mse_loss(ground_truth_images, em_pred_images)}")
    print(f"Moment matching: {mse_loss(ground_truth_images, mm_pred_images)}")
    print(f"Linearization: {mse_loss(ground_truth_images, linearized_pred_images)}")
    print("------------------------------------------------------------------------------------")

    print("------------------------------------------------------------------------------------")
    print("NLPD (t=64):")
    print(f"Euler-Maruyama: {calculate_gaussian_nlpd(z0, pred_gp_euler_mean[-1], em_lst_cov)}")
    print(f"Moment matching: {calculate_gaussian_nlpd(z0, ode_sigma_euler_m[-1], ode_sigma_euler_P[-1])}")
    print(f"Linearization: {calculate_gaussian_nlpd(z0, ode_linearized_euler_m[-1], ode_linearized_euler_P[-1])}")
    print("------------------------------------------------------------------------------------")
