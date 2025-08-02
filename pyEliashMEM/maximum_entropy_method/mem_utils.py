import numpy as np
from scipy.linalg import eigh, lu_factor, lu_solve
from typing import Tuple


def calc_score(A: np.array, M: np.array) -> np.array:
    """
    Computes the entropy-based score (relative entropy or Kullback–Leibler divergence) between model and prior.

    This function calculates a score that quantifies the divergence between the current model `A` and a prior model `M`,
    commonly used in maximum entropy methods (MEM). The expression used is:

        S = Σ [ A - M + A · log(A / M) ]

    Parameters:
        A (np.ndarray): Current model coefficients (length NA).
        M (np.ndarray): Prior model coefficients (same shape as `A`).

    Returns:
        float: Entropy-based score measuring the deviation from the prior.

    Raises:
        ValueError: If `A` and `M` have different shapes.
        FloatingPointError: If `A` or `M` contains zeros or negative values (log or divide-by-zero).

    Notes:
        - This is equivalent to the negative of the Kullback–Leibler divergence between `M` and `A` (up to constants).
        - Used in MEM to balance fit quality with adherence to prior information.
    """
    S = np.sum(A - M + A * np.log(A/M))
    return S


def setup_ktk(ND: int, KERN: np.ndarray, SIGMA: np.ndarray) -> np.array:
    """
    Computes the weighted kernel matrix product Kᵀ W K for spectral fitting.

    This function calculates the matrix product of the transpose of the kernel matrix `KERN`
    with itself, weighted by the inverse variance from `SIGMA`. This is a key step in
    constructing the normal equations for weighted least squares or MEM fitting.

    Parameters:
        ND (int): Number of data points to consider (rows of `KERN` and length of `SIGMA`).
        KERN (np.ndarray): Kernel matrix of shape (ND, NA).
        SIGMA (np.ndarray): Standard deviations (uncertainties) associated with each data point (length ≥ ND).

    Returns:
        np.ndarray: The weighted kernel product matrix `Kᵀ W K` of shape (NA, NA).

    Raises:
        ValueError: If dimensions of `KERN` or `SIGMA` are incompatible with `ND`.
        ZeroDivisionError: If any value in `SIGMA[:ND]` is zero (division by zero).

    Notes:
        - Weight matrix `W` is diagonal with elements `1 / SIGMA[i]^2`.
        - This weighted product is essential in MEM and other weighted fitting techniques.
    """
    inv_sigma2 = 1.0 / SIGMA[:ND] ** 2  # (ND,)
    weighted_KERN = KERN[:ND, :] * inv_sigma2[:, np.newaxis]  # shape (ND, NA)

    # Compute K^T W K
    KTK = np.dot(KERN[:ND].T, weighted_KERN)

    return KTK


def setup_ktd(ND: int, KERN: np.ndarray, D: np.ndarray, SIGMA: np.ndarray) -> np.array:
    """
    Computes the weighted kernel matrix product Kᵀ W K for spectral fitting.

    This function calculates the matrix product of the transpose of the kernel matrix `KERN`
    with itself, weighted by the inverse variance from `SIGMA`. This is a key step in
    constructing the normal equations for weighted least squares or MEM fitting.

    Parameters:
        ND (int): Number of data points to consider (rows of `KERN` and length of `SIGMA`).
        KERN (np.ndarray): Kernel matrix of shape (ND, NA).
        SIGMA (np.ndarray): Standard deviations (uncertainties) associated with each data point (length ≥ ND).

    Returns:
        np.ndarray: The weighted kernel product matrix `Kᵀ W K` of shape (NA, NA).

    Raises:
        ValueError: If dimensions of `KERN` or `SIGMA` are incompatible with `ND`.
        ZeroDivisionError: If any value in `SIGMA[:ND]` is zero (division by zero).

    Notes:
        - Weight matrix `W` is diagonal with elements `1 / SIGMA[i]^2`.
        - This weighted product is essential in MEM and other weighted fitting techniques.
    """
    inv_sigma2 = 1.0 / SIGMA[:ND]**2  # (ND,)
    weighted_D = D[:ND] * inv_sigma2  # (ND,)
    KTD = np.dot(KERN[:ND].T, weighted_D)

    return KTD


def skilling_itr(NA, KTK, KTD, M, A, ALPHA) -> Tuple[np.array, np.array]:
    """
    Performs one iteration of Skilling’s algorithm for maximum entropy model update.

    This function updates the model vector `A` by solving a constrained quadratic
    optimization problem that balances fitting the data (`KTD`, `KTK`) and
    maintaining closeness to the prior `M`, regulated by `ALPHA`.

    The update uses an expansion in a 3-dimensional subspace defined by entropy
    gradients and residuals, solving a small linear system for optimal step size
    while ensuring positivity of `A`.

    Parameters:
        NA (int): Number of model parameters.
        KTK (np.ndarray): Kernel matrix product `Kᵀ W K` (shape NA × NA).
        KTD (np.ndarray): Kernel-data product `Kᵀ W D` (length NA).
        M (np.ndarray): Prior model vector (length NA).
        A (np.ndarray): Current model estimate (length NA).
        ALPHA (float): Regularization parameter controlling entropy strength.

    Returns:
        tuple:
            - A (np.ndarray): Updated model vector after this iteration.
            - DA (float): Norm of the update step, useful as a convergence metric.

    Raises:
        LinAlgError: If the linear system `Q @ V = V` is singular or ill-conditioned.
        ValueError: If input arrays have incompatible shapes.

    Notes:
        - Ensures updated `A` remains positive by limiting step size `ETA`.
        - Implements a trust-region like step constrained by positivity.
        - Based on Skilling’s maximum entropy iterative fitting scheme.
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
    V = np.linalg.solve(Q, V)

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
    Performs an iterative Newton update for the regularization parameter ALPHA in MEM.

    This function updates the entropy regularization parameter `ALPHA` by solving
    a nonlinear equation derived from the balance between the entropy term and
    data fit, using eigenvalues of the weighted kernel matrix.

    Parameters:
        NA (int): Number of model parameters.
        KTK (np.ndarray): Kernel matrix product `Kᵀ W K` (shape NA × NA).
        M (np.ndarray): Prior model vector (length NA).
        A (np.ndarray): Current model estimate (length NA).
        ALPHA (float): Initial guess for the regularization parameter.

    Returns:
        tuple:
            - ALPHA (float): Updated regularization parameter after iteration.
            - DALPHA (float): Absolute change in `ALPHA` during the iteration.

    Raises:
        LinAlgError: If eigenvalue decomposition fails.
        RuntimeWarning: If convergence criteria are not met within iteration loop.

    Notes:
        - Uses eigenvalues of the matrix sqrt(A) * KTK * sqrt(A) for update.
        - Newton method is damped with a factor of 0.1 for stability.
        - Iteration stops when relative change in `ALPHA` is less than 1e-6.
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
    Computes the error matrix for the maximum entropy method via matrix inversion.

    This function constructs and inverts the matrix (Kᵀ W K + ALPHA * diag(1/A)),
    which combines data fidelity and entropy regularization terms. The inverse
    matrix provides error estimates for the spectral coefficients.

    Parameters:
        NA (int): Number of model parameters.
        KTK (np.ndarray): Kernel matrix product `Kᵀ W K` (shape NA × NA).
        A (np.ndarray): Current model vector (length NA), assumed strictly positive.
        ALPHA (float): Regularization parameter.

    Returns:
        tuple:
            - DDQ (np.ndarray): Regularized matrix (shape NA × NA) = KTK + ALPHA * diag(1/A).
            - DADA (np.ndarray): Inverse of `DDQ`, representing the error covariance matrix.

    Raises:
        LinAlgError: If matrix inversion fails due to singularity or ill-conditioning.
        ValueError: If any element of `A` is zero or negative (division by zero).

    Notes:
        - Uses LU decomposition and solve for inversion, analogous to LAPACK DGESV.
        - The diagonal regularization ensures stability and positivity constraints.
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


def chi(KERN, D, SIGMA, A):
    """
    Computes the chi-squared statistic measuring the goodness of fit.

    This function calculates the chi-squared value between the model prediction
    (KERN @ A) and observed data `D`, weighted by the uncertainties `SIGMA`.

    Parameters:
        KERN (np.ndarray): Kernel matrix mapping model to data (shape: data points × model parameters).
        D (np.ndarray): Observed data vector.
        SIGMA (np.ndarray): Standard deviations (uncertainties) of the data points.
        A (np.ndarray): Model coefficients vector.

    Returns:
        float: Chi-squared statistic quantifying weighted squared residuals.

    Raises:
        ValueError: If shapes of `KERN`, `D`, `SIGMA`, and `A` are incompatible.
        ZeroDivisionError: If any value in `SIGMA` is zero (division by zero).
    """
    # Compute TEMP = KERN @ A - D
    TEMP = KERN @ A - D

    # Compute chi-squared: sum((TEMP / SIGMA)^2)
    CHI = np.sum((TEMP / SIGMA) ** 2)

    return CHI