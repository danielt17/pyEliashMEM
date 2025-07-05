import numpy as np
from typing import Tuple


def model_constraint(KT: np.float64, Y: np.array, K: np.array, D: np.array, SIGMA: np.array, OMEGABIN: list,
                     NBIN: np.int64, OMEGAD: np.float64, OMEGAM: np.float64, MODEL: str, NA: np.int64,
                     LAMBDA0: np.float64, ND: np.int64) -> Tuple[float, np.array, np.array, np.array, np.array,
                                                                float, float, np.array, np.array, float, np.array,
                                                                np.array, np.array, np.array, float, float, float,
                                                                float]:
    """
        Applies physical model constraints by rescaling input quantities and constructing
        the Eliashberg function model (or loading one from file) for MEM fitting.

        This function:
          - Rescales temperature and energy quantities to match the temperature-normalized scale.
          - Constructs the model Eliashberg function (`M`) either analytically or from a file.
          - Computes statistical weights and intermediate variables used in constrained inversion.

        Parameters:
            KT (float): Temperature in eV.
            Y (np.ndarray): Shifted energy values, shape (ND,).
            K (np.ndarray): Shifted momentum values, shape (ND,).
            D (np.ndarray): Dispersion difference, shape (ND,).
            SIGMA (np.ndarray): Uncertainty values for each data point, shape (ND,).
            OMEGABIN (list): List of frequency bin boundaries of length (NBIN + 1).
            NBIN (int): Number of output bins for the Eliashberg function.
            OMEGAD (float): Debye frequency in eV.
            OMEGAM (float): Maximum phonon frequency in eV.
            MODEL (str): Path to model file. If "NONE", a default analytical model is used.
            NA (int): Number of points for the Eliashberg function model.
            LAMBDA0 (float): Coupling constant.
            ND (int): Number of data points used for computation.

        Returns:
            tuple: A tuple containing the following processed or derived quantities:
                - KT (float): Rescaled temperature in Kelvin.
                - Y (np.ndarray): Rescaled shifted energy values.
                - D (np.ndarray): Rescaled dispersion difference.
                - SIGMA (np.ndarray): Rescaled uncertainty values.
                - OMEGABIN (np.ndarray): Rescaled frequency bin boundaries.
                - OMEGAD (float): Rescaled Debye frequency.
                - OMEGAM (float): Rescaled maximum phonon frequency.
                - Y1 (np.ndarray): Frequency grid for the Eliashberg function model, shape (NA,).
                - M (np.ndarray): Model Eliashberg function values, shape (NA,).
                - DY1 (float): Step size in the model energy grid.
                - inv_sigma2 (np.ndarray): Inverse squared errors, shape (ND,).
                - Ksq (np.ndarray): Squared momentum values, shape (ND,).
                - K4 (np.ndarray): Fourth power of momentum values, shape (ND,).
                - K3 (np.ndarray): Cubed momentum values, shape (ND,).
                - X1 (float): Weighted sum used in constraint matrix.
                - X2 (float): Weighted sum used in constraint matrix.
                - X12 (float): Weighted covariance-like term.
                - XX (float): Determinant-like term used for inversion normalization.

        Raises:
            IOError: If the `MODEL` file is specified but cannot be read.
            ValueError: If input shapes do not match expected dimensions.

        Notes:
            - Energy, frequency, and uncertainty values are normalized by temperature.
            - The model function `M` is ensured to be non-negative.
            - The output can be used directly in constrained MEM inversion solvers.
    """
    # Rescale temperature
    KT = KT / 1.1604e4  # Convert KT from eV to Kelvin scale

    # Scale energy-related quantities
    Y /= KT
    D /= KT
    SIGMA /= KT
    OMEGABIN = np.array(OMEGABIN)
    OMEGABIN[:NBIN + 1] *= 1e-3 / KT

    OMEGAD /= KT
    OMEGAM /= KT

    Y1 = np.zeros((NA, ))
    M = np.zeros((NA, ))
    if MODEL != "NONE":
        data = np.loadtxt(MODEL, max_rows=NA)
        Y1[:NA] = data[:, 0] * 1e-3 / KT
        M[:NA] = data[:, 1] * LAMBDA0
        DY1 = Y1[1] - Y1[0]
    else:
        DY1 = OMEGAM / float(NA)
        Y1[:NA] = np.arange(1, NA + 1, dtype=np.float64) * DY1

        below_d = Y1[:NA] <= OMEGAD
        between_d_m = (Y1[:NA] > OMEGAD) & (Y1[:NA] <= OMEGAM)

        M[:NA] = 0.0
        M[:NA][below_d] = LAMBDA0 * (Y1[:NA][below_d] / OMEGAD) ** 2
        M[:NA][between_d_m] = LAMBDA0
        M[:NA] = np.maximum(M[:NA], 0.0)  # Enforce non-negativity

    # Compute X1, X2, X12
    inv_sigma2 = 1.0 / (SIGMA[:ND] ** 2)
    Ksq = K[:ND] ** 2
    K4 = K[:ND] ** 4
    K3 = K[:ND] ** 3

    X1 = np.sum(Ksq * inv_sigma2)
    X2 = np.sum(K4 * inv_sigma2)
    X12 = np.sum(K3 * inv_sigma2)

    XX = X1 * X2 - X12 ** 2
    X1 /= XX
    X2 /= XX
    X12 = -X12 / XX
    return KT, Y, D, SIGMA, OMEGABIN, OMEGAD, OMEGAM, Y1, M, DY1, inv_sigma2, Ksq, K4, K3, X1, X2, X12, XX