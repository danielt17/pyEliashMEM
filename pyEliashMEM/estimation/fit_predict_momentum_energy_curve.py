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


def estimate_error(ND: np.int64, E: np.ndarray, D: np.ndarray, params: dict) -> np.ndarray:
    """
    Computes the uncertainty (SIGMA) for each data point based on energy and
    dispersion difference values.

    If `ERRB0` is non-positive, a local second-order polynomial fit is performed
    in a sliding window to estimate error. If `ERRB0` is positive, a linear
    error model is used.

    Parameters:
        ND (np.int64): Number of data points.
        E (np.ndarray): Energy values of length ND (in eV). Will be negated internally.
        D (np.ndarray): Dispersion difference values of length ND.
        params (dict): Dictionary containing at least:
            - 'ERRB0' (float): Constant component of the error.
            - 'ERRB1' (float): Linear coefficient for energy-dependent error.

    Returns:
        np.ndarray: An array of length ND containing estimated uncertainty values (SIGMA)
                    for each data point.

    Raises:
        np.linalg.LinAlgError: If a singular matrix is encountered during local fitting.

    Notes:
        - Uses a sliding window of size NBIN = 9 for local polynomial fitting.
        - If `ERRB0 == 0.0`, the function assigns a uniform error based on the mean squared
          error from all valid local fits.
        - The energy array `E` is internally flipped (multiplied by -1) to match the
          behavior in the original Fortran implementation.
    """
    NBIN = 9
    SIGMA = np.zeros(ND)
    E = -E  # Flip the energy axis

    if params["ERRB0"] <= 0.0:
        for i in range(ND):
            il = i - NBIN // 2
            if il < 0:
                il = 0
            if il + NBIN > ND:
                il = ND - NBIN

            QD = D[il:il + NBIN].copy()
            Q = np.vstack([E[il:il + NBIN] ** k for k in range(3)]).T  # shape (NBIN, 3)

            QTQ = Q.T @ Q
            QTD = Q.T @ QD

            try:
                coeffs = np.linalg.solve(QTQ, QTD)
                residuals = Q @ coeffs - QD
                SIGMA[i] = np.linalg.norm(residuals) / np.sqrt(NBIN - 3)
            except np.linalg.LinAlgError:
                SIGMA[i] = np.nan  # If singular matrix

        sigmabar = np.sqrt(np.mean(SIGMA[~np.isnan(SIGMA)] ** 2))

        if params["ERRB0"] == 0.0:
            SIGMA[:] = sigmabar

        print(f"SIGMABAR           = {sigmabar:.6g}")
    else:
        SIGMA = params["ERRB0"] + params["ERRB1"] * E

    return SIGMA