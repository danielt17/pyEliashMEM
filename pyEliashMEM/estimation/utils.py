import numpy as np
from pyEliashMEM.utils.params import Constants


def f(x):
    if x >= 0:
        return np.exp(-x) / (np.exp(-x) + 1.0)
    else:
        return 1.0 / (np.exp(x) + 1.0)


def nb(x):
    return np.exp(-x) / (1.0 - np.exp(-x))


def setup_kernel(ND: int, NA: int, Y: np.array, Y1: np.array, DY1: np.array) -> np.array:
    """
        Computes the kernel matrix KERN of shape (ND, NA) used in further calculations.

        The kernel is calculated by summing terms of an infinite series until convergence,
        for each pair of points (Y[i], Y1[j]) over the indices i and j.

        Parameters
        ----------
        ND : int
            Number of points in Y.
        NA : int
            Number of points in Y1.
        Y : np.ndarray
            1D array of length ND, containing energy or frequency values.
        Y1 : np.ndarray
            1D array of length NA, containing energy or frequency values.
        DY1 : float
            Scalar step size related to Y1, used to scale the kernel values.

        Returns
        -------
        np.ndarray
            Kernel matrix of shape (ND, NA) computed as per the specified formula.
            Each element KERN[i, j] corresponds to the sum over N of terms involving
            Y[i] and Y1[j], scaled by DY1.

        Notes
        -----
        The series summation for each element terminates when the absolute value of the
        current term GN is less than 1e-6 times the absolute value of the cumulative sum G,
        ensuring numerical convergence.
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
    Compute IMSIGMA = π * DY1 * sum_i [AF[i] * (F(Y1[i] - Y) + F(Y1[i] + Y) + 2 * NB(Y1[i]))]

    Parameters:
        NA   (int): Length of input arrays
        Y    (float): Scalar value
        AF   (ndarray): Array of weights, shape (NA,)
        Y1   (ndarray): Array of positions, shape (NA,)
        DY1  (float): Spacing

    Returns:
        float: IMSIGMA result
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
    Compute frequency-bin weighted statistics.

    Parameters:
        NA       (int): Length of A, Y1
        NBIN     (int): Number of bins
        OMEGABIN (ndarray): Bin edges, length NBIN+1
        BETA     (float): Exponent
        A        (ndarray): Weights array, shape (NA,)
        Y1       (ndarray): Support points, shape (NA,)
        DY1      (float): Spacing
        EM       (ndarray): Error matrix, shape (NA, NA)

    Returns:
        EBX  (ndarray): Bin centers, shape (NBIN,)
        EBY  (ndarray): Weighted sum in bin, shape (NBIN,)
        EBDX (ndarray): Bin half-widths, shape (NBIN,)
        EBDY (ndarray): Error estimate in bin, shape (NBIN,)
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


def intavg(NA, A, Y1, DY, EM):
    """
    Compute LAMBDA, DLAMBDA, and OMEGALOG from integral averages.

    Parameters:
        NA      (int): Length of arrays
        A       (ndarray): Weight array, shape (NA,)
        Y1      (ndarray): Support array, shape (NA,)
        DY      (float): Step size
        EM      (ndarray): Error covariance matrix, shape (NA, NA)

    Returns:
        LAMBDA     (float): 2 * sum(A / Y1) * DY
        DLAMBDA    (float): sqrt of propagated uncertainty
        OMEGALOG   (float): exp(2 * DY / LAMBDA * sum(A / Y1 * log(Y1)))
    """

    # Compute LAMBDA
    LAMBDA = 2.0 * DY * np.sum(A / Y1)

    # Compute DLAMBDA
    Y1_inv = 1.0 / Y1
    outer_inv = np.outer(Y1_inv, Y1_inv)
    DLAMBDA = 2.0 * DY * np.sqrt(np.sum(EM * outer_inv))

    # Compute OMEGALOG
    OMEGALOG = np.exp(2.0 * DY / LAMBDA * np.sum((A / Y1) * np.log(Y1)))

    return LAMBDA, DLAMBDA, OMEGALOG