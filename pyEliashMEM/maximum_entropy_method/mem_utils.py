import numpy as np
from scipy.linalg import eigh, lu_factor, lu_solve
from typing import Tuple


def calc_score(A: np.array, M: np.array) -> np.array:
    S = np.sum(A - M + A * np.log(A/M))
    return S


def setup_ktk(ND: int, KERN: np.ndarray, SIGMA: np.ndarray) -> np.array:
    """
    Computes the KTK matrix used in maximum entropy fitting:
    KTK = KERN.T @ diag(1/SIGMA**2) @ KERN

    Parameters
    ----------
    ND : int
        Number of data points.
    NA : int
        Number of spectral/momentum points.
    KERN : np.ndarray of shape (ND, NA)
        Kernel matrix.
    SIGMA : np.ndarray of shape (ND,)
        Standard deviations of the measurements.
    KTK : np.ndarray of shape (NA, NA)
        Output: matrix to store the result K^T W K.
        Modified in place.
    """
    inv_sigma2 = 1.0 / SIGMA[:ND] ** 2  # (ND,)
    weighted_KERN = KERN[:ND, :] * inv_sigma2[:, np.newaxis]  # shape (ND, NA)

    # Compute K^T W K
    KTK = np.dot(KERN[:ND].T, weighted_KERN)

    return KTK


def setup_ktd(ND: int, KERN: np.ndarray, D: np.ndarray, SIGMA: np.ndarray) -> np.array:
    """
    Computes the KTD vector used in maximum entropy fitting:
    KTD = KERN.T @ (D / SIGMA^2)

    Parameters
    ----------
    ND : int
        Number of data points.
    NA : int
        Number of spectral/momentum points.
    KERN : np.ndarray of shape (ND, NA)
        Kernel matrix.
    D : np.ndarray of shape (ND,)
        Data residual vector.
    SIGMA : np.ndarray of shape (ND,)
        Standard deviations of the measurements.
    KTD : np.ndarray of shape (NA,)
        Output: vector to store the result K^T W D.
        Modified in place.
    """
    inv_sigma2 = 1.0 / SIGMA[:ND]**2  # (ND,)
    weighted_D = D[:ND] * inv_sigma2  # (ND,)
    KTD = np.dot(KERN[:ND].T, weighted_D)

    return KTD


def skilling_itr(NA, KTK, KTD, M, A, ALPHA) -> Tuple[np.array, np.array]:
    """
    Python equivalent of Fortran subroutine SKILLING_ITR
    Modifies A in-place and returns DA (norm of update step)
    """
    # Initialize arrays
    E0 = np.zeros((NA, 3), dtype=np.float64)
    E = np.zeros((NA, 3), dtype=np.float64)
    TEMP = np.zeros(NA, dtype=np.float64)
    Q = np.zeros((3, 3), dtype=np.float64)
    V = np.zeros(3, dtype=np.float64)

    # Compute E0(:, 0) = -log(A / M)
    E0[:, 0] = -np.log(A / M)
    NRME1 = np.linalg.norm(E0[:, 0])

    # E0(:, 1) = KTD - KTK @ A
    E0[:, 1] = KTD.copy()
    E0[:, 1] = KTK @ A - E0[:, 1]
    NRME2 = np.linalg.norm(E0[:, 1])

    # TEMP = -E0(:, 1) + ALPHA * E0(:, 0)
    TEMP = -E0[:, 1] + ALPHA * E0[:, 0]
    DA = np.linalg.norm(TEMP)

    # TEMP = (E0[:,0]/NRME1 - E0[:,1]/NRME2) * A
    TEMP = (E0[:, 0] / NRME1 - E0[:, 1] / NRME2) * A

    # E0(:, 2) = KTK @ TEMP
    E0[:, 2] = KTK @ TEMP

    # E = A * E0
    for j in range(3):
        E[:, j] = A * E0[:, j]

    # TEMP = E0[:,1] - ALPHA * E0[:,0]
    TEMP = E0[:, 1] - ALPHA * E0[:, 0]

    # V = - dot(TEMP, E[:, j])
    for j in range(3):
        V[j] = -np.dot(TEMP, E[:, j])

    # Compute Q matrix
    for j in range(3):
        TEMP = ALPHA * E0[:, j]
        TEMP += KTK @ E[:, j]
        for i in range(j, 3):
            Q[i, j] = np.dot(E[:, i], TEMP)
            Q[j, i] = Q[i, j]

    # Solve Q @ V = V (overwrites V)
    V = np.linalg.pinv(Q) @ V

    # TEMP = linear combination of E[:, i] with coefficients V[i]
    TEMP.fill(0.0)
    for i in range(3):
        TEMP += V[i] * E[:, i]

    # Find minimum ETA to keep A positive
    ETAMIN = 1.0
    for i in range(NA):
        if TEMP[i] < 0.0:
            ETA = -0.5 * A[i] / TEMP[i]
            if ETA < ETAMIN:
                ETAMIN = ETA

    # Update A
    A += ETAMIN * TEMP

    return A, DA


def alpha_itr(NA, KTK, M, A, ALPHA):
    """
    Python version of the ALPHA_ITR Fortran subroutine.

    Parameters:
    - NA     : int
    - KTK    : (NA, NA) ndarray
    - M, A   : (NA,) ndarray
    - ALPHA  : float (initial alpha)

    Returns:
    - ALPHA  : updated alpha
    - DALPHA : absolute change in alpha
    """

    # Store initial ALPHA
    ALPHA0 = ALPHA

    # Construct AKTKA = sqrt(A) * KTK * sqrt(A)
    sqrt_A = np.sqrt(A)
    AKTKA = KTK * sqrt_A[:, None] * sqrt_A[None, :]

    # Compute eigenvalues of AKTKA (symmetric matrix)
    LAMBDA = eigh(AKTKA, eigvals_only=True, turbo=True, check_finite=False, lower=False)

    # Compute entropy-related term S
    S = np.sum(A - M - A * np.log(A / M))

    # Iterative Newton update for ALPHA
    DAL = ALPHA
    while abs(DAL) > 1e-6 * ALPHA:
        F = 2.0 * ALPHA * S + np.sum(LAMBDA / (ALPHA + LAMBDA))
        DF = 2.0 * S - np.sum(LAMBDA / (ALPHA + LAMBDA) ** 2)

        DAL = -0.1 * F / DF
        ALPHA += DAL

    DALPHA = abs(ALPHA - ALPHA0)
    return ALPHA, DALPHA


def error_matrix(NA, KTK, A, ALPHA):
    """
    Compute DADA = inv(KTK + diag(ALPHA / A))

    Parameters:
        NA     (int): Size of matrices
        KTK    (ndarray): Input square matrix of shape (NA, NA)
        A      (ndarray): Vector of length NA
        ALPHA  (float): Scalar regularization parameter

    Returns:
        DDQ    (ndarray): Regularized matrix
        DADA   (ndarray): Inverse of DDQ
    """

    # Step 1: Copy KTK into DDQ
    DDQ = KTK.copy()

    # Step 2: Modify diagonal of DDQ: DDQ[i, i] += ALPHA / A[i]
    DDQ[np.diag_indices(NA)] += ALPHA / A

    # Step 3: Initialize DADA to identity matrix
    DADA = np.eye(NA)

    # Step 4: Solve DDQ * DADA = I using LU decomposition (equivalent to DGESV)
    lu, piv = lu_factor(DDQ)
    DADA = lu_solve((lu, piv), DADA)

    return DDQ, DADA


import numpy as np


def chi(KERN, D, SIGMA, A):
    """
    Compute chi-squared error:
        chi = sum_i [ ( (KERN @ A - D)[i] )^2 / SIGMA[i]^2 ]

    Parameters:
        ND     (int): Number of data points
        NA     (int): Number of coefficients
        KERN   (ndarray): ND x NA kernel matrix
        D      (ndarray): ND data vector
        SIGMA  (ndarray): ND vector of standard deviations
        A      (ndarray): NA coefficient vector

    Returns:
        chi2   (float): Chi-squared value
    """

    # Compute TEMP = KERN @ A - D
    TEMP = KERN @ A - D

    # Compute chi-squared: sum((TEMP / SIGMA)^2)
    CHI = np.sum((TEMP / SIGMA) ** 2)

    return CHI