import numpy as np
from pyEliashMEM.utils.params import Constants


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


def f(x):
    if x >= 0:
        return np.exp(-x) / (np.exp(-x) + 1.0)
    else:
        return 1.0 / (np.exp(x) + 1.0)


def nb(x):
    return np.exp(-x) / (1.0 - np.exp(-x))
