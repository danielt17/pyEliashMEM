import numpy as np
from typing import Tuple

def fit_predict_momentum_energy_curve(eraw: np.array,
                                      kraw: np.array,
                                      params: dict,
                                      config: dict
                                      ) -> Tuple[np.array, np.array, np.array, np.array]:
    """
    Fits and predicts the momentum-energy curve from raw (shifted) data, using
    either manual parameters or a least-squares regression, depending on config.

    The function:
      - Applies a mask to select energy values within ECUTOFF and less than 0.
      - If estimation is enabled, performs a least-squares fit of the form:
            E ≈ A1·k + A2·k²
      - If estimation is disabled, uses A1 and A2 provided in `params`.
      - Computes the predicted curve and estimation error.

    Parameters:
        eraw (np.ndarray): Array of energy values (E - EF).
        kraw (np.ndarray): Array of momentum values (k - KF).
        params (dict): Dictionary of physical and fitting parameters, must include:
            - 'ECUTOFF' (float): Maximum absolute energy for inclusion.
            - 'A1' (float): Linear coefficient (used if estimation is disabled).
            - 'A2' (float): Quadratic coefficient (used if estimation is disabled).
        config (dict): Configuration dictionary with estimation control flags:
            - enables -> enb_estimations -> enb_all_estimations (bool)
            - enables -> enb_estimations -> enb_estimation_momentum_energy_curve (bool)

    Returns:
        tuple:
            - predicted_curve (np.ndarray): Estimated energy values over all kraw (full prediction).
            - Y (np.ndarray): Filtered energy values (within cutoff and < 0).
            - D (np.ndarray): Residuals between Y and predicted values at same points.
            - K (np.ndarray): Filtered momentum values corresponding to Y.

    Notes:
        - The estimation is a simple linear regression in the basis [k, k²].
        - If estimation is disabled, prediction uses provided A1 and A2.
    """
    if config["enables"]["enb_estimations"]["enb_all_estimations"] and config["enables"]["enb_estimations"]["enb_estimation_momentum_energy_curve"]:
        X = np.vstack([kraw, kraw**2]).T
        coeffs, residuals, rank, s = np.linalg.lstsq(X, eraw, rcond=None)
        A1, A2 = coeffs
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
