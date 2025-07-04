# imports
import os
from pyEliashMEM.utils.read_inputs import read_and_prepare_data
from pyEliashMEM.plots.plot_momentum_energy_curve import plot_momentum_energy_curve
from pyEliashMEM.estimation.fit_predict_momentum_energy_curve import fit_predict_momentum_energy_curve

def main():
    config, params, eraw, kraw, output_folder = read_and_prepare_data()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plot_momentum_energy_curve(eraw, kraw, params, config, output_folder)
    Y, D, K = fit_predict_momentum_energy_curve(eraw, kraw, params, config)
    pass


if __name__ == "__main__":
    main()
