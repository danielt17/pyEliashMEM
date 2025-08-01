import numpy as np
from pyEliashMEM.maximum_entropy_method.mem_utils import setup_ktk, setup_ktd, skilling_itr, alpha_itr, error_matrix


def memfit_cls(ND, NA, ITERNUM, KERN, D, SIGMA, M):
    """
    Performs MEM fitting using the Constrained Least Squares (CLS) method.

    Parameters
    ----------
    ND : int
        Number of data points.
    NA : int
        Number of alpha points.
    ITERNUM : int
        Maximum number of MEM iterations.
    KERN : np.ndarray
        Kernel matrix of shape (ND, NA).
    D : np.ndarray
        Data difference vector (ND,).
    SIGMA : np.ndarray
        Standard deviation for each point (ND,).
    M : np.ndarray
        Default model (NA,).
    A : np.ndarray
        Output: spectral function (NA,) — modified in-place.
    DA_container : list
        Output: single-element list to hold final DA value (so it's mutable).
    ALPHA_container : list
        Output: single-element list to hold final ALPHA value (so it's mutable).
    DALPHA_container : list
        Output: single-element list to hold final DALPHA value (so it's mutable).
    EM : np.ndarray
        Output: error matrix (NA, NA).

    Notes
    -----
    - This function modifies `A`, `DA_container`, `ALPHA_container`, `DALPHA_container`, and `EM` in place.
    - The logic requires the following external subroutines to be implemented:
        - setup_ktk
        - setup_ktd
        - skilling_itr
        - alpha_itr
        - error_matrix
    """

    # Compute KTK and KTD
    KTK = setup_ktk(ND, KERN, SIGMA)
    KTD = setup_ktd(ND, KERN, D, SIGMA)

    # Initialize A with slight perturbation of M
    A = (1.0 + 1e-6) * M

    DA = 1.0
    DALPHA = 1.0
    ALPHA = 1.0
    iteration = 0

    # Iteratively update A and ALPHA
    while (DA > 1e-8 or DALPHA > 1e-8) and iteration <= ITERNUM:
        A, DA = skilling_itr(NA, KTK, KTD, M, A, ALPHA)
        ALPHA, DALPHA = alpha_itr(NA, KTK, M, A, ALPHA)
        iteration += 1
        print(f"DA: {DA}, ALPHA: {ALPHA}, DALPHA: {DALPHA}")

    # Compute error matrix
    DDQ, EM = error_matrix(NA, KTK, A, ALPHA)

    return A, DA, KERN, SIGMA, ALPHA, DALPHA, EM