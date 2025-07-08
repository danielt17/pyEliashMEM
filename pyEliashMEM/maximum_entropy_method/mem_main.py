import numpy as np
from typing import Tuple
from pyEliashMEM.maximum_entropy_method.mem_algos import memfit_cls


def iterative_mem_fit(A1, A2, ND: int, NA: int, ITERNUM: int, METHOD: int, FITBPD: int, KERN: np.array, D: np.array,
                      SIGMA: np.array, M: np.array, ALPHA: float, DALPHA: float, XCHI: float,
                      Y: np.array, K: np.array, KT: np.float, X1: float, X2: float, X12: float) -> \
                      Tuple[np.array, np.array, np.array, int]:
    """
    Performs an iterative maximum entropy fitting procedure,
    updating A1 and A2 until convergence or a maximum iteration count is reached.

    Parameters
    ----------
    ND : int
        Number of data points.
    NA : int
        Number of model points.
    ITERNUM : int
        Number of internal MEM iterations per fit.
    METHOD : int
        MEM method selector (1: HST, 2: CLS, 3: BRYAN, 4: FIXALPHA).
    FITBPD : int
        Flag to enable band dispersion fitting.
    KERN : np.ndarray
        Kernel matrix of shape (ND, NA).
    D : np.ndarray
        Residual dispersion array of shape (ND,).
    SIGMA : np.ndarray
        Error bar array of shape (ND,).
    M : np.ndarray
        Default model of shape (NA,).
    A : np.ndarray
        Output spectral function of shape (NA,).
    DA : np.ndarray
        Delta A vector (NA,).
    ALPHA : float
        Initial alpha regularization parameter.
    DALPHA : float
        Change in alpha for fixed-alpha methods.
    XCHI : float
        Target chi-squared value.
    EM : np.ndarray
        Temporary working array for MEM.
    Y : np.ndarray
        Energy array (ND,).
    K : np.ndarray
        Momentum array (ND,).
    KT : float
        Rescaled temperature (in eV).
    X1, X2, X12 : float
        Precomputed constants for momentum-energy fit.

    Returns
    -------
    tuple
        A1, A2, D, and number of outer iterations.
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
            memfit_cls(ND, NA, ITERNUM, KERN, D, SIGMA, M, ALPHA, DALPHA)
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

    return A1, A2, D, J
