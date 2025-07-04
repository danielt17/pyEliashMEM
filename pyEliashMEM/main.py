# imports
import os
from pyEliashMEM.utils.read_inputs import read_and_prepare_data
from pyEliashMEM.plots.plot_momentum_energy_curve import plot_momentum_energy_curve, plot_momentum_energy_curve_with_fit
from pyEliashMEM.estimation.fit_predict_momentum_energy_curve import fit_predict_momentum_energy_curve


def main():
    config, params, eraw, kraw, output_folder = read_and_prepare_data()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plot_momentum_energy_curve(eraw, kraw, params, config, output_folder)
    predicted_curve, Y, D, K = fit_predict_momentum_energy_curve(eraw, kraw, params, config)
    plot_momentum_energy_curve_with_fit(eraw, kraw, predicted_curve, params, config, output_folder)
    pass


if __name__ == "__main__":
    main()
