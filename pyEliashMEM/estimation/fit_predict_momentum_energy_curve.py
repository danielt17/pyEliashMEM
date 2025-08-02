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
            - ND (np.int64): total number of elements post calculation.
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
    ND = np.sum(mask)
    Y = eraw[mask]
    K = kraw[mask]
    predicted_curve = A1 * kraw + A2 * kraw ** 2
    # Estimation error
    D = Y - predicted_curve[mask]
    return predicted_curve, ND, Y, D, K


def estimate_error(ND: np.int32, E: np.ndarray, D: np.ndarray, params: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the uncertainty (SIGMA) for each data point using either local polynomial
    fitting or a linear error model.

    If `ERRB0 <= 0`, the function performs a local second-order polynomial fit in a sliding
    window of size 9 over the data to compute residual-based errors. If `ERRB0 > 0`, it uses
    a linear model `SIGMA = ERRB0 + ERRB1 * E`.

    Parameters:
        ND (np.int64): Number of data points.
        E (np.ndarray): Energy values of shape (ND,). Internally flipped in sign.
        D (np.ndarray): Dispersion difference values of shape (ND,).
        params (dict): Dictionary containing error model parameters:
            ERRB0 (float): Constant component of the error.
            ERRB1 (float): Linear coefficient for energy-dependent error.

    Returns:
        np.ndarray: Array of shape (ND,) containing estimated uncertainties (SIGMA) for each point.

    Raises:
        np.linalg.LinAlgError: If a singular matrix occurs during local polynomial fitting.

    Notes:
        - Uses a fixed window size of NBIN = 9 for local least-squares fitting.
        - Energy values are internally negated to match the Fortran implementation logic.
        - If any least-squares problem is ill-conditioned, SIGMA for that point is set to NaN.
        - If ERRB0 == 0, all SIGMA values are replaced by the RMS average (SIGMABAR) of successful fits.
    """
    NBIN = 9
    pad = NBIN // 2
    SIGMA = np.zeros(ND)
    E = -E  # Flip energy axis as per Fortran behavior

    if params["ERRB0"] <= 0.0:
        # Generate all window start indices with clamping at edges
        il_array = np.clip(np.arange(ND) - pad, 0, ND - NBIN)

        # Create an array of shape (ND, NBIN) containing window indices
        idx_matrix = il_array[:, None] + np.arange(NBIN)[None, :]  # shape (ND, NBIN)

        # Gather E and D in windowed form (ND, NBIN)
        E_windows = E[idx_matrix]
        D_windows = D[idx_matrix]

        # Polynomial design matrix: (ND, NBIN, 3)
        Q = np.stack([E_windows ** k for k in range(3)], axis=-1)

        # Compute normal equations
        QT = np.transpose(Q, (0, 2, 1))  # (ND, 3, NBIN)
        QTQ = QT @ Q  # (ND, 3, 3)
        QTD = QT @ D_windows[..., None]  # (ND, 3, 1)

        # Solve least squares for each window
        coeffs = np.empty((ND, 3, 1))
        failed = np.zeros(ND, dtype=bool)

        for i in range(ND):
            try:
                coeffs[i] = np.linalg.solve(QTQ[i], QTD[i])
            except np.linalg.LinAlgError:
                failed[i] = True

        # Compute residuals
        fit = np.einsum('ijk,ikl->ijl', Q, coeffs).squeeze(-1)  # (ND, NBIN)
        residuals = fit - D_windows
        residual_norm = np.linalg.norm(residuals, axis=1)

        SIGMA[~failed] = residual_norm[~failed] / np.sqrt(NBIN - 3)
        SIGMA[failed] = np.nan

        sigmabar = np.sqrt(np.nanmean(SIGMA ** 2))

        if params["ERRB0"] == 0.0:
            SIGMA[:] = sigmabar

        print(f"SIGMABAR           = {sigmabar:.6g}")
    else:
        SIGMA = params["ERRB0"] + params["ERRB1"] * E

    return SIGMA, E