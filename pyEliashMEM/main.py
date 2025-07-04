# imports
import os
from pyEliashMEM.utils.read_inputs import read_and_prepare_data
from pyEliashMEM.plots.plot_momentum_energy_curve import plot_momentum_energy_curve


def main():
    config, params, eraw, kraw = read_and_prepare_data()
    plot_momentum_energy_curve(eraw, kraw, params, config)
    pass


if __name__ == "__main__":
    main()
