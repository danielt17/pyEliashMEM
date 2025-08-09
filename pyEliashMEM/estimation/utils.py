import numpy as np
from pyEliashMEM.utils.params import Constants


def f(x: np.float64):
    """
    Computes the Fermi-Dirac distribution function.

    This function evaluates the occupation probability for fermions at
    dimensionless energy `x = (E - μ) / kT`, ensuring numerical stability
    across positive and negative values of `x`.

    Parameters:
        x (float): Dimensionless energy variable.

    Returns:
        float: Occupation number according to the Fermi-Dirac distribution.

    Notes:
        - For `x >= 0`, the function avoids overflow in `exp(x)` by rewriting
          the expression to use `exp(-x)` instead.
        - This stable form is particularly useful for large |x| values.

    Raises:
        OverflowError: If `x` is a large negative number and `exp(x)` overflows
                       (very rare in practice due to use of safe formulation).
    """
    if x >= 0:
        return np.exp(-x) / (np.exp(-x) + 1.0)
    else:
        return 1.0 / (np.exp(x) + 1.0)


def nb(x: np.array):
    """
    Computes the Bose-Einstein distribution function.

    This function evaluates the Bose-Einstein occupation number for a given
    dimensionless energy `x = E / kT`.

    Parameters:
        x (float or np.ndarray): Dimensionless energy (energy divided by temperature, E / kT).

    Returns:
        float or np.ndarray: Occupation number according to the Bose-Einstein distribution.

    Raises:
        ZeroDivisionError: If `x` is exactly zero (division by zero).
        OverflowError: If `x` is a large negative number, leading to numerical overflow.
    """
    return np.exp(-x) / (1.0 - np.exp(-x))


def setup_kernel(ND: int, NA: int, Y: np.array, Y1: np.array, DY1: np.array) -> np.array:
    """
    Constructs the kernel matrix for the spectral fitting problem.

    This function builds a kernel `K[i, j]` used to relate model coefficients
    to observable quantities in the spectral model. Each kernel element is
    computed via a convergent infinite series involving the input energies `Y`
    and model grid `Y1`. The summation is truncated adaptively when terms
    become sufficiently small.

    Parameters:
        ND (int): Number of data points (length of `Y`).
        NA (int): Number of model coefficients (length of `Y1`).
        Y (np.ndarray): Observed energy values (length ND).
        Y1 (np.ndarray): Model energy grid (length NA).
        DY1 (np.ndarray): Differential element associated with `Y1` grid.

    Returns:
        np.ndarray: Kernel matrix of shape (ND, NA), where each element is computed as:
            K[i, j] = sum over N of a rational function involving `Y[i]` and `Y1[j]`.

    Notes:
        - The kernel is built using a summation over Matsubara-like terms:
            G_N ~ Y[i] * Y1[j] / ((Y[i] ± Y1[j])² + [(2N+1)π]²)
        - The summation stops when the next term becomes smaller than
          1e-6 times the current total, to ensure convergence.

    Raises:
        ValueError: If `Y`, `Y1`, or `DY1` have incompatible shapes.
        AttributeError: If `Constants.PI2` is not defined (must be set to π²).
    """
    KERN = np.zeros((ND, NA), dtype=np.float64)
    for j in range(NA):
        for i in range(ND):
            G = 0.0
            GN = 1.0
            N = 0
            while abs(GN) > 1e-6 * abs(G if G != 0 else 1):
                term_num = 8 * Constants.PI2 * (2 * N + 1) * Y[i] * Y1[j]
                denom1 = (Y[i] - Y1[j]) ** 2 + ((2 * N + 1) ** 2) * Constants.PI2
                denom2 = (Y[i] + Y1[j]) ** 2 + ((2 * N + 1) ** 2) * Constants.PI2
                GN = term_num / denom1 / denom2
                G += GN
                N += 1
            KERN[i, j] = G * DY1
    return KERN


def IMSIGMA(NA, Y, AF, Y1, DY1):
    """
    Computes the imaginary part of the self-energy (ImΣ) at a given energy `Y`.

    This function evaluates the spectral broadening or scattering rate using
    a sum over model coefficients, incorporating contributions from both
    the Fermi-Dirac and Bose-Einstein distribution functions.

    Parameters:
        NA (int): Number of model coefficients.
        Y (float): Energy at which the imaginary part is evaluated.
        AF (np.ndarray): Model coefficient array (length NA).
        Y1 (np.ndarray): Model energy grid (length NA).
        DY1 (float): Grid spacing of `Y1` (assumed uniform).

    Returns:
        float: The imaginary part of the self-energy at energy `Y`.

    Notes:
        - The formula used is:
            ImΣ(Y) = π * DY1 * Σ_i [ AF[i] * (f(Y1[i] - Y) + f(Y1[i] + Y) + 2 * nb(Y1[i])) ]
        - `f` is the Fermi-Dirac distribution function.
        - `nb` is the Bose-Einstein distribution function.
        - This form ensures proper thermal broadening and detailed balance.

    Raises:
        ValueError: If the length of `AF` or `Y1` does not match `NA`.
    """
    summation = 0.0
    for i in range(NA):
        summation += AF[i] * (
                f(Y1[i] - Y) + f(Y1[i] + Y) + 2.0 * nb(Y1[i])
        )

    IMSIGMA = np.pi * DY1 * summation
    return IMSIGMA


def weight(NA, NBIN, OMEGABIN, BETA, A, Y1, DY1, EM):
    """
    Computes binned spectral weight and statistical uncertainty measures.

    This function bins a spectral function defined by coefficients `A` over energy grid `Y1`
    into the bin edges defined by `OMEGABIN`. It calculates both the weighted integral (EBY)
    and an associated uncertainty (EBDY) in each bin. The binned quantities are raised to the power `BETA`.

    Parameters:
        NA (int): Number of spectral coefficients.
        NBIN (int): Number of output bins (must satisfy len(OMEGABIN) == NBIN + 1).
        OMEGABIN (np.ndarray): Bin edges for energy (length NBIN + 1).
        BETA (float): Power used in weighting (`Y1^BETA`).
        A (np.ndarray): Spectral weight coefficients (length NA).
        Y1 (np.ndarray): Energy grid corresponding to `A` (length NA).
        DY1 (float): Spacing between grid points in `Y1`.
        EM (np.ndarray): Covariance matrix of the spectral coefficients (NA × NA).

    Returns:
        tuple:
            - EBX (np.ndarray): Bin centers (length NBIN).
            - EBY (np.ndarray): Binned and weighted spectral sum over each bin (length NBIN).
            - EBDX (np.ndarray): Half-widths of each bin (length NBIN).
            - EBDY (np.ndarray): Estimated uncertainty in each bin (length NBIN).

    Raises:
        ValueError: If array shapes are inconsistent, or `OMEGABIN` length != NBIN + 1.

    Notes:
        - The bin center is computed as: EBX[i] = 0.5 * (OMEGABIN[i+1] + OMEGABIN[i])
        - The integral in each bin is: EBY[i] = DY1 * Σ_j A[j] * Y1[j]^BETA
        - The uncertainty is estimated using the covariance matrix:
            EBDY[i] = sqrt(DY1^2 * Σ_{j,k in bin} EM[j,k] * (Y1[j]Y1[k])^BETA)
    """

    EBX = np.zeros(NBIN)
    EBY = np.zeros(NBIN)
    EBDX = np.zeros(NBIN)
    EBDY = np.zeros(NBIN)

    L = 0
    while L < NA and Y1[L] < OMEGABIN[0]:
        L += 1

    for i in range(NBIN):
        EBX[i] = 0.5 * (OMEGABIN[i+1] + OMEGABIN[i])
        EBDX[i] = 0.5 * (OMEGABIN[i+1] - OMEGABIN[i])

        L0 = L
        while L < NA and Y1[L] < OMEGABIN[i+1]:
            EBY[i] += A[L] * Y1[L]**BETA
            L += 1
        L1 = L - 1

        EBY[i] *= DY1

        EBDY_sum = 0.0
        for j in range(L0, L1 + 1):
            for k in range(L0, L1 + 1):
                EBDY_sum += EM[j, k] * (Y1[j] * Y1[k])**BETA
        EBDY[i] = np.sqrt(EBDY_sum) * DY1

    return EBX, EBY, EBDX, EBDY


def intavg(A, Y1, DY, EM):
    """
    Computes average spectral quantities: mean inverse energy, its uncertainty, and a logarithmic mean.

    This function evaluates integrated spectral averages using model weights `A`, the energy grid `Y1`,
    and the covariance matrix `EM`. It returns:
      1. `LAMBDA`: A weighted mean inverse energy.
      2. `DLAMBDA`: The uncertainty in `LAMBDA` using the error matrix `EM`.
      3. `OMEGALOG`: A logarithmic energy average derived from the distribution of `A`.

    Parameters:
        A (np.ndarray): Spectral weights (length NA).
        Y1 (np.ndarray): Energy grid (length NA).
        DY (float): Energy grid spacing.
        EM (np.ndarray): Covariance matrix of the weights (shape NA × NA).

    Returns:
        tuple:
            - LAMBDA (float): Weighted mean inverse energy: LAMBDA = 2·DY·Σ (A / Y1)
            - DLAMBDA (float): Uncertainty in `LAMBDA`, estimated from `EM`.
            - OMEGALOG (float): Logarithmic spectral mean based on weighting A/Y1.

    Raises:
        ValueError: If `A`, `Y1`, or `EM` have incompatible shapes.
        ZeroDivisionError: If any entry in `Y1` is zero (division by zero).
        RuntimeWarning: If `Y1` contains negative or zero values leading to invalid logs.

    Notes:
        - The quantity `OMEGALOG` is calculated via:
              OMEGALOG = exp[(2·DY / LAMBDA) * Σ (A / Y1) * log(Y1)]
        - This is useful in MEM-like spectral reconstructions to characterize distribution centroids.
    """

    # Compute LAMBDA
    LAMBDA = 2.0 * DY * np.sum(A / Y1)

    # Compute DLAMBDA
    Y1_inv = 1.0 / Y1
    outer_inv = np.outer(Y1_inv, Y1_inv)
    DLAMBDA = 2.0 * DY * np.sqrt(np.sum(EM * outer_inv))

    # Compute OMEGALOG
    # Our OMEGALOG is correct while fortran is wrong! they have a bug in log computation.
    # specifically np.log(0.3868)=-0.9498475154029978 which is correct in python
    # but fortran says it is LOG(0.3868)=-2.5726781331127264
    # so here we have the correct value
    OMEGALOG = np.exp(2.0 * DY / LAMBDA * np.sum((A / Y1) * np.log(Y1)))

    return LAMBDA, DLAMBDA, OMEGALOG