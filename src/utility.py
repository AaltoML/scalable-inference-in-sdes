import numpy as np


def get_observations(exp_name="circle", n_observations=8):
    """
    Get observations for the bean or circle experiment
    """
    u = v = yu = yv = None
    if exp_name not in ["bean", "circle"]:
        print("Currently only bean and circle experiments are supported")
        return u, v, yu, yv

    if exp_name == "circle":
        u = 0.5 * np.cos(np.linspace(-np.pi, np.pi, n_observations))
        v = 0.5 * np.sin(np.linspace(-np.pi, np.pi, n_observations))

        yu = np.sin(np.linspace(-np.pi, np.pi, n_observations))
        yv = -np.cos(np.linspace(-np.pi, np.pi, n_observations))

    elif exp_name == "bean":
        a = 1
        th = np.linspace(0, np.pi, 63).T
        r = a * np.sin(th) ** 3 + a * np.cos(th) ** 3
        u = r * np.cos(th) - 0.3
        v = r * np.sin(th) - 0.3

        yu = np.diff(u)
        yv = np.diff(v)

        n = 63//n_observations

        u = u[::n]
        v = v[::n]

        yu = yu[::n] * 10
        yv = yv[::n] * 10

    return u, v, yu, yv


def get_ellipsoid(mean, covariance):
    """
    The function takes mean and covariance as input and return the ellipsoid parameters.
    """
    eigen_value, eigen_vector = np.linalg.eig(covariance)

    largest_eigen_val_loc = np.argmax(eigen_value)
    largest_eigen_vec = eigen_vector[:, largest_eigen_val_loc]
    largest_eigen_val = eigen_value[largest_eigen_val_loc]

    smallest_eigen_val_loc = np.argmin(eigen_value)
    smallest_eigen_val = eigen_value[smallest_eigen_val_loc]

    angle = np.arctan2(largest_eigen_vec[1], largest_eigen_vec[0])
    if angle < 0:
        angle = angle + 2 * np.pi
    angle = np.degrees(angle)

    chisquare_val = 2.4477

    major_axis = 2 * chisquare_val * np.sqrt(largest_eigen_val)
    minor_axis = 2 * chisquare_val * np.sqrt(smallest_eigen_val)

    return mean, major_axis, minor_axis, angle


def get_trajectory_euler(initial_state, v_model, n_trajectories=100, end_state_t=50, dt=1.):
    """
    Get trajectories using Euler-Maruyama.
    """
    previous_state = np.repeat(
        np.array(initial_state, dtype=np.float64).reshape((1, 2)),
        n_trajectories,
        axis=0,
    )

    all_paths = previous_state
    all_paths = all_paths.reshape((-1, 1, 2))

    n_states = int(end_state_t / dt)

    mvn = np.random.multivariate_normal
    for s in range(1, n_states):
        pred_m, pred_S = v_model.predict_mean_var(previous_state, full_cov=True)
        pred_m = pred_m.reshape(previous_state.shape)
        pred_S = pred_S.reshape((-1, previous_state.shape[0], previous_state.shape[0]))

        x_pred_m = pred_m[:, 0]
        y_pred_m = pred_m[:, 1]
        x_pred_S = pred_S[0, :, :].reshape((previous_state.shape[0], -1))
        y_pred_S = pred_S[1, :, :].reshape((previous_state.shape[0], -1))

        x_pred_covar = np.diag(
            np.diag(x_pred_S.reshape((n_trajectories, n_trajectories)))
        )
        y_pred_covar = np.diag(
            np.diag(y_pred_S.reshape((n_trajectories, n_trajectories)))
        )

        x_pred = mvn(x_pred_m.reshape(-1), x_pred_covar, 1).reshape(n_trajectories)
        y_pred = mvn(y_pred_m.reshape(-1), y_pred_covar, 1).reshape(n_trajectories)

        previous_state = previous_state + dt * np.stack(
            (x_pred, y_pred), axis=1
        ).reshape(-1, 2)

        all_paths = np.hstack((all_paths, previous_state.reshape((-1, 1, 2))))

    return all_paths
