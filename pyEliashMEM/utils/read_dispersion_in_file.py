import numpy as np


def read_dispersion_data(filename):
    data = np.loadtxt(filename)
    eraw, kraw = data[:, 0], data[:, 1]
    return eraw, kraw


def shift_dispersion_data(eraw, kraw, params):
    eraw_shifted = eraw - params["EF"]
    kraw_shifted = kraw - params["KF"]
    return eraw_shifted, kraw_shifted


def read_and_shift_dispersion_data(filename, params):
    eraw, kraw = read_dispersion_data(filename)
    eraw, kraw = shift_dispersion_data(eraw, kraw, params)
    return eraw, kraw



