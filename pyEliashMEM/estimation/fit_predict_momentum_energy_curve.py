import numpy as np


def fit_predict_momentum_energy_curve(eraw: np.array, kraw: np.array, params: dict, config: dict) -> (np.array,
                                                                                                      np.array,
                                                                                                      np.array,
                                                                                                      np.array):
    """
        Filters and computes a derived curve from momentum-energy data, optionally
        performing a fitting/estimation based on the configuration.

        The function:
          - Applies a mask to select energies within cutoff and less than zero.
          - Computes a dependent variable D = Y - A1 * K - A2 * K^2 using parameters, the estimation error.
          - Placeholder for future fitting/estimation if enabled in config.

        Parameters:
            eraw (np.ndarray): Array of raw energy values.
            kraw (np.ndarray): Array of raw momentum values.
            params (dict): Dictionary of parameters, must include:
                - 'ECUTOFF' (float): Energy cutoff threshold.
                - 'A1' (float): Linear coefficient for momentum.
                - 'A2' (float): Quadratic coefficient for momentum.
            config (dict): Configuration dictionary with estimation flags under:
                - 'enables' -> 'enb_estimations' -> 'enb_all_estimations' (bool)
                - 'enables' -> 'enb_estimations' -> 'enb_estimation_momentum_energy_curve' (bool)

        Returns:
            tuple:
                - predicted_curve (np.array): the fitted prediction
                - Y (np.ndarray): Filtered energy values satisfying cutoff and negativity.
                - D (np.ndarray): Computed dependent variable array, estimation of fit error.
                - K (np.ndarray): Filtered momentum values matching Y.

        Notes:
            - Currently, the estimation branch (when enabled) is a placeholder (`pass`).
            - If estimation is disabled, parameters A1 and A2 from `params` are used directly.
    """
    if config["enables"]["enb_estimations"]["enb_all_estimations"] and config["enables"]["enb_estimations"]["enb_estimation_momentum_energy_curve"]:
        X = np.vstack([kraw, kraw**2]).T
        coeffs, residuals, rank, s = np.linalg.lstsq(X, eraw, rcond=None)
        A1, A2 = coeffs
        pass
    else:
        A1 = params["A1"]
        A2 = params["A2"]
    mask = (np.abs(eraw) <= params["ECUTOFF"]) & (eraw < 0)
    Y = eraw[mask]
    K = kraw[mask]
    predicted_curve = A1 * kraw + A2 * kraw ** 2
    # Estimation error
    D = Y - predicted_curve[mask]
    return predicted_curve, Y, D, K
