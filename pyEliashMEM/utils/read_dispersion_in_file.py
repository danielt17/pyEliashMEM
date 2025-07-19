import numpy as np
from typing import Tuple

def read_dispersion_data(filename: str, params: dict) -> Tuple[np.array, np.array]:
    """
        Reads raw dispersion data from a text file.

        The file should contain two columns:
            - Column 1: energy values (e.g., E(k))
            - Column 2: momentum values (e.g., k)

        Parameters:
            filename (str): Path to the dispersion data file.

        Returns:
            tuple:
                - eraw (np.ndarray): Array of raw energy values.
                - kraw (np.ndarray): Array of raw momentum values.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is invalid or cannot be parsed as float values.
    """
    data = np.loadtxt(filename)
    eraw, kraw = data[:params["NDRAW"], 0], data[:params["NDRAW"], 1]
    return eraw, kraw


def shift_dispersion_data(eraw: np.array, kraw: np.array, params: dict) -> Tuple[np.array, np.array]:
    """
    Applies energy and momentum shifts to the raw dispersion data using EF and KF.

    This is typically done to align the data with the Fermi energy and Fermi wavevector.

    Parameters:
        eraw (np.ndarray): Array of raw energy values.
        kraw (np.ndarray): Array of raw momentum values.
        params (dict): Dictionary containing at least:
            - 'EF' (float): Fermi energy to subtract from each energy value.
            - 'KF' (float): Fermi momentum to subtract from each momentum value.

    Returns:
        tuple:
            - eraw_shifted (np.ndarray): Energy values shifted by EF.
            - kraw_shifted (np.ndarray): Momentum values shifted by KF.

    Raises:
        KeyError: If 'EF' or 'KF' is missing in the `params` dictionary.
    """

    eraw_shifted = eraw - params["EF"]
    kraw_shifted = kraw - params["KF"]
    return eraw_shifted, kraw_shifted


def read_and_shift_dispersion_data(filename: str, params: dict) -> Tuple[np.array, np.array]:
    """
        Reads raw dispersion data from a file and applies EF/KF shifts.

        This function combines two steps:
          1. Load energy and momentum data from a file.
          2. Shift the data by subtracting the Fermi energy (EF) and Fermi momentum (KF),
             as specified in the `params` dictionary.

        Parameters:
            filename (str): Path to the file containing raw dispersion data.
            params (dict): Dictionary containing shift parameters:
                - 'EF' (float): Fermi energy.
                - 'KF' (float): Fermi momentum.

        Returns:
            tuple:
                - eraw (np.ndarray): Energy values shifted by EF.
                - kraw (np.ndarray): Momentum values shifted by KF.

        Raises:
            FileNotFoundError: If the dispersion file does not exist.
            KeyError: If 'EF' or 'KF' is missing from `params`.
            ValueError: If the data file is malformed or contains non-numeric entries.
    """
    eraw, kraw = read_dispersion_data(filename, params)
    eraw, kraw = shift_dispersion_data(eraw, kraw, params)
    return eraw, kraw



