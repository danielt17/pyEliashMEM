import numpy as np


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