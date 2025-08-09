import numpy as np
from typing import Tuple
from dataclasses import dataclass
from pyEliashMEM.maximum_entropy_method.mem_algos import memfit_cls
from pyEliashMEM.maximum_entropy_method.mem_utils import setup_ktk, chi, calc_score
from pyEliashMEM.estimation.utils import setup_kernel, IMSIGMA, weight, intavg


def iterative_mem_fit(A1, A2, ND: int, NA: int, ITERNUM: int, METHOD: int, FITBPD: int, KERN: np.array, D: np.array,
                      SIGMA: np.array, M: np.array, ALPHA: float, DALPHA: float, XCHI: float,
                      Y: np.array, K: np.array, KT: np.float, X1: float, X2: float, X12: float,
                      dispersion_data_output: dataclass) -> \
                      Tuple[np.array, np.array, np.array, int, dataclass]:
    """
    Performs iterative optimization of dispersion parameters A1 and A2 using MEM fitting.

    This function refines the coefficients of a quadratic dispersion model through repeated
    maximum entropy method (MEM) fits. Depending on the specified method, it invokes a
    different MEM fitting routine and iteratively updates A1 and A2 based on the fit results.

    The update is guided by minimizing the misfit between the model and observed data,
    taking into account a prior model and regularization.

    Parameters:
        A1 (float): Initial guess for linear dispersion coefficient.
        A2 (float): Initial guess for quadratic dispersion coefficient.
        ND (int): Number of data points.
        NA (int): Number of model parameters.
        ITERNUM (int): Number of iterations for each MEM fit.
        METHOD (int): Integer specifying the MEM method to use:
            - 1: HST method
            - 2: Classic MEM
            - 3: Bryan's method
            - 4: Fixed-alpha method
        FITBPD (int): Flag for whether to iteratively update the dispersion model (1: yes, 0: no).
        KERN (np.ndarray): Kernel matrix for data transformation.
        D (np.ndarray): Current data misfit vector (will be updated in-place).
        SIGMA (np.ndarray): Uncertainty values associated with each data point.
        M (np.ndarray): Prior model used in MEM fitting.
        ALPHA (float): Initial regularization coefficient.
        DALPHA (float): Initial regularization step size.
        XCHI (float): Target chi-squared value for MEM convergence.
        Y (np.ndarray): Observed data vector.
        K (np.ndarray): Momentum values corresponding to each data point.
        KT (float): Inverse temperature (1 / kT), used in rescaling.
        X1, X2, X12 (float): Precomputed factors used in linear system update of A1, A2.

    Returns:
        tuple:
            - A1 (float): Final optimized linear dispersion coefficient.
            - A2 (float): Final optimized quadratic dispersion coefficient.
            - D (np.ndarray): Final data misfit vector.
            - J (int): Number of iterations performed.
            - A (np.ndarray): Final solution vector from MEM fit.
            - DA (np.ndarray): Final uncertainty or change in A.
            - KERN (np.ndarray): Final kernel matrix (may be updated in MEM).
            - SIGMA (np.ndarray): Possibly updated uncertainties from MEM.
            - ALPHA (float): Final regularization coefficient.
            - DALPHA (float): Final regularization delta.
            - EM (np.ndarray): Final energy mesh/grid used in MEM fitting.

    Raises:
        ValueError: If an unsupported MEM method is specified.
        RuntimeError: If iteration exceeds `max_iter` without convergence (implicit).
    """
    A1 = -A1
    A2 = -A2
    DA1, DA2 = 1.0, 1.0
    max_iter = 200
    J = 0

    while (DA1 > 1e-3 or DA2 > 1e-2) and J <= max_iter:
        J += 1

        # Call appropriate MEM fit routine
        if METHOD == 1:
            memfit_hst(ND, NA, ITERNUM, KERN, D, SIGMA, M, ALPHA, XCHI)
        elif METHOD == 2:
            A, DA, KERN, SIGMA, ALPHA, DALPHA, EM = memfit_cls(ND, NA, ITERNUM, KERN, D, SIGMA, M)
        elif METHOD == 3:
            memfit_bryan(ND, NA, ITERNUM, KERN, D, SIGMA, M, ALPHA, DALPHA)
        elif METHOD == 4:
            memfit_fixalpha(ND, NA, ITERNUM, KERN, D, SIGMA, M, ALPHA, DALPHA)
        else:
            raise ValueError("Unsupported MEM method")

        if FITBPD == 0:
            break  # goto 1000 in Fortran

        # Compute D1 = KERN @ A
        D1 = KERN @ A

        # Update S1 and S2
        weighted = KT * (Y + D1) / (SIGMA ** 2)
        S1 = np.sum(weighted * K)
        S2 = np.sum(weighted * K ** 2)

        # Store previous A1, A2
        prev_A1, prev_A2 = A1, A2

        # Update A1, A2
        A1 = X2 * S1 + X12 * S2
        A2 = X12 * S1 + X1 * S2

        # Compute convergence deltas
        DA1 = abs(prev_A1 - A1) / abs(A1) if A1 != 0 else 0
        DA2 = abs(prev_A2 - A2) / abs(A2) if A2 != 0 else 0

        # Update D
        D[:] = (A1 * K + A2 * K ** 2) / KT - Y

        print(f"Iteration {J}: A1 = {A1:.6g}, A2 = {A2:.6g}")

    A1 = -A1
    A2 = -A2

    dispersion_data_output.a1_est = A1
    dispersion_data_output.a2_est = A2
    dispersion_data_output.dalpha = DALPHA

    return A1, A2, D, J, A, DA, KERN, SIGMA, ALPHA, DALPHA, EM, dispersion_data_output


def score_output(params: dict, KERN: np.array, D: np.array, SIGMA: np.array, A: np.array, M: np.array,
                 ALPHA: np.float64, ND: np.int32, Y: np.float64, Y1: np.array, DY1: np.float64, OMEGABIN: np.array,
                 EM: np.array, dispersion_data_output: dataclass) -> Tuple[np.float64, np.float64, np.float64, np.array,
                                                                           np.array, np.array, np.array, np.array,
                                                                           np.array, np.float64, np.float64, np.float64,
                                                                           dataclass]:
    """
    Computes key quantities related to spectral fitting and optimization.

    This function performs multiple steps required in spectral model evaluation:
      1. Calculates the chi-squared-like misfit term (CHI0) using a kernel and other parameters.
      2. Computes a model score (S) and a regularized objective function (Q).
      3. Applies the kernel to the model (D1).
      4. Estimates the imaginary part of the self-energy (IMS) for each data point.
      5. Calculates weighted error components (EBX, EBY, EBDX, EBDY).
      6. Computes spectral averages (LAMBDA, DLAMBDA) and the logarithmic omega grid (OMEGALOG).

    Parameters:
        params (dict): Dictionary of model and solver parameters. Must contain:
            - 'NA' (int): Number of model coefficients.
            - 'NBIN' (int): Number of energy bins.
            - 'BETA' (float): Inverse temperature or regularization parameter.
        KERN (np.ndarray): Kernel matrix used in the chi computation.
        D (np.ndarray): Data vector or measurement.
        SIGMA (np.ndarray): Error or noise vector.
        A (np.ndarray): Model coefficients.
        M (np.ndarray): Prior model or reference.
        ALPHA (float): Regularization coefficient.
        ND (int): Number of data points.
        Y (np.ndarray): Data values.
        Y1 (np.ndarray): Model output corresponding to Y.
        DY1 (np.ndarray): Uncertainties or errors in Y1.
        OMEGABIN (np.ndarray): Energy bin grid for spectral averaging.
        EM (np.ndarray): Energy mesh/grid for spectral integration.

    Returns:
        tuple:
            - CHI0 (float): Misfit term from the chi calculation.
            - S (float): Model score.
            - Q (float): Regularized objective value.
            - D1 (np.ndarray): Model transformed by the kernel.
            - IMS (np.ndarray): Imaginary part of self-energy for each data point.
            - EBX, EBY, EBDX, EBDY (np.ndarray): Weighted error contributions in X and Y directions.
            - LAMBDA (float): Weighted average of the spectrum.
            - DLAMBDA (float): Standard deviation of the spectrum.
            - OMEGALOG (np.ndarray): Logarithmic energy grid.

    Raises:
        KeyError: If required keys ('NA', 'NBIN', 'BETA') are missing from `params`.
        ValueError: If array shapes are inconsistent or inputs are malformed.
    """
    CHI0 = chi(KERN, D, SIGMA, A)
    S = calc_score(A, M)
    Q = CHI0 / 2 - ALPHA * S
    D1 = KERN @ A
    IMS = np.empty(ND)
    for i in range(ND):
        IMS[i] = IMSIGMA(params["NA"], Y[i], A, Y1, DY1)
    EBX, EBY, EBDX, EBDY = weight(params["NA"], params["NBIN"], OMEGABIN, params["BETA"], A, Y1, DY1, EM)
    LAMBDA, DLAMBDA, OMEGALOG = intavg(A, Y1, DY1, EM)

    dispersion_data_output.chi2 = CHI0
    dispersion_data_output.q = Q
    dispersion_data_output.alpha = ALPHA
    dispersion_data_output.lambda_ = LAMBDA
    dispersion_data_output.d_lambda = DLAMBDA
    dispersion_data_output.omega_log = OMEGALOG

    return CHI0, S, Q, D1, IMS, EBX, EBY, EBDX, EBDY, LAMBDA, DLAMBDA, OMEGALOG, dispersion_data_output


def dispersion_output(params: dict, KT: np.float64, eraw: np.array, Y1: np.array, DY1: np.float64, A: np.array,
                      A1: np.float64, A2: np.float64) -> Tuple[np.array, np.array, np.array, np.array, np.array,
                      np.array]:
    """
    Processes dispersion data and computes renormalized momentum values.

    This function performs several steps related to electronic dispersion modeling:
      1. Rescales the raw energy data using the inverse temperature (KT).
      2. Constructs a kernel matrix based on the rescaled energy and model output.
      3. Applies the kernel to the model coefficients to obtain the shifted data (D1).
      4. Computes a renormalized momentum (K) based on a quadratic dispersion relation.
      5. Estimates the imaginary part of the self-energy (IMS) at each energy point.

    Parameters:
        params (dict): Dictionary containing model parameters. Must include:
            - 'NDRAW' (int): Number of dispersion data points.
            - 'NA' (int): Number of model coefficients.
        KT (float): Inverse temperature (1 / kT) used to scale energies.
        eraw (np.ndarray): Raw energy values to be scaled.
        Y1 (np.ndarray): Model-predicted observable corresponding to each energy.
        DY1 (np.ndarray): Uncertainties associated with `Y1`.
        A (np.ndarray): Model coefficient array.
        A1 (float): Linear coefficient in the dispersion relation.
        A2 (float): Quadratic coefficient in the dispersion relation.

    Returns:
        tuple:
            - eraw (np.ndarray): Rescaled energy values (in units of KT).
            - KERN (np.ndarray): Kernel matrix constructed from scaled energy and model.
            - D1 (np.ndarray): Model-transformed energy shift.
            - K (np.ndarray): Renormalized momentum values.
            - IMS (np.array): Imaginary part of self-energy at the last energy point.
            - FWHM (np.array): photoemission FWHM
    Raises:
        KeyError: If required keys ('NDRAW', 'NA') are missing from `params`.
        ValueError: If inputs have incompatible dimensions or invalid types.
    """
    eraw *= -1.0 / KT
    KERN = setup_kernel(params["NDRAW"], params["NA"], eraw, Y1, DY1)
    D1 = KERN @ A
    K = np.zeros(params["NDRAW"])
    IMS = np.zeros(params["NDRAW"],)
    FWHM = np.zeros(params["NDRAW"], )
    for i in range(params["NDRAW"]):
        E0 = - (eraw[i] + D1[i]) * KT
        denominator = np.abs(A1) + np.sqrt(A1 ** 2 + 4.0 * A2 * E0)
        K[i] = 2.0 * E0 / denominator * np.sign(A1)
        IMS[i] = IMSIGMA(params["NA"], eraw[i], A, Y1, DY1)
        FWHM[i] = 2 * IMS[i] * KT / np.abs(A1+2*A2*K[i])
    return eraw, KERN, D1, K, IMS, FWHM