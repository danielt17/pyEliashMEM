# imports
import os
import pandas as pd
from pyEliashMEM.utils.format_data_output import DispersionData
from pyEliashMEM.utils.read_inputs import read_and_prepare_data
from pyEliashMEM.plots.plot_momentum_energy_curve import plot_momentum_energy_curve, plot_momentum_energy_curve_with_fit
from pyEliashMEM.plots.plot_eliashberg_function import plot_eliashberg_function_curve_with_constraint
from pyEliashMEM.plots.plot_self_energy import plot_self_energy_real_part, plot_self_energy_imaginary_part, \
    plot_real_part_self_energy_and_eliashberg_function
from pyEliashMEM.estimation.fit_predict_momentum_energy_curve import fit_predict_momentum_energy_curve, estimate_error
from pyEliashMEM.estimation.constraints import model_constraint
from pyEliashMEM.estimation.utils import setup_kernel
from pyEliashMEM.maximum_entropy_method.mem_main import iterative_mem_fit, score_output, dispersion_output


def main():
    dispersion_data_output = DispersionData()
    config, params, eraw, kraw, output_folder, dispersion_data_output = read_and_prepare_data(dispersion_data_output)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plot_momentum_energy_curve(eraw, kraw, params, config, output_folder)
    predicted_curve, ND, Y, D, K, dispersion_data_output = fit_predict_momentum_energy_curve(eraw, kraw, params,
                                                                                             config,
                                                                                             dispersion_data_output)
    plot_momentum_energy_curve_with_fit(eraw, kraw, predicted_curve, params, config, output_folder)
    SIGMA, Y, dispersion_data_output = estimate_error(ND, Y, D, params, dispersion_data_output)
    KT, Y, D, SIGMA, OMEGABIN, OMEGAD, OMEGAM, Y1, M, DY1, inv_sigma2, Ksq, K4, K3, X1, X2, X12, XX = \
        model_constraint(params["KT"], Y, K, D, SIGMA, params["OMEGABIN"], params["NBIN"], params["OMEGAD"],
                         params["OMEGAM"], params["MODEL"], params["NA"], params["LAMBDA0"], ND)
    KERN = setup_kernel(ND, params["NA"], Y, Y1, DY1)
    A1, A2, D, J, A, DA, KERN, SIGMA, ALPHA, DALPHA, EM, dispersion_data_output = \
        iterative_mem_fit(params["A1"], params["A2"], ND, params["NA"], params["ITERNUM"], params["METHOD"],
                          params["FITBPD"], KERN, D, SIGMA, M, params["ALPHA"], params["DALPHA"], params["XCHI"], Y, K,
                          KT, X1, X2, X12, dispersion_data_output)
    eliashberg_function = pd.DataFrame({
        "omega[meV]": Y1 * KT * 1000,
        "Eliashberg function": A,
        "constraint function": M
    })
    plot_eliashberg_function_curve_with_constraint(eliashberg_function, config, output_folder)
    if config["enables"]["enb_saves"]["enb_all_saves"] and config["enables"]["enb_saves"]["enb_save_logs"]:
        eliashberg_function.to_csv(os.path.join(output_folder, "eliashberg_function.csv"), index=False)
    CHI0, S, Q, D1, IMS, EBX, EBY, EBDX, EBDY, LAMBDA, DLAMBDA, OMEGALOG, dispersion_data_output = \
        score_output(params, KERN, D, SIGMA, A, M, ALPHA, ND, Y, Y1, DY1, OMEGABIN, EM, dispersion_data_output)
    self_energy = pd.DataFrame({
        "omega[meV]": Y * KT * 1000,
        "real part of self energy[meV]": D * KT * 1000,
        "error bars[meV]": SIGMA * KT * 1000,
        "fit real part of self energy[meV]": D1 * KT * 1000,
        "calculated imaginary part of self energy[meV]": IMS * KT * 1000
    })
    plot_self_energy_real_part(self_energy, config, output_folder)
    plot_self_energy_imaginary_part(self_energy, config, output_folder)
    plot_real_part_self_energy_and_eliashberg_function(self_energy, eliashberg_function, config, output_folder)
    if config["enables"]["enb_saves"]["enb_all_saves"] and config["enables"]["enb_saves"]["enb_save_logs"]:
        self_energy.to_csv(os.path.join(output_folder, "self_energy.csv"), index=False)
    eraw, KERN, D1, K, IMS, FWHM = dispersion_output(params, KT, eraw, Y1, DY1, A, A1, A2)
    dispersion_fit = pd.DataFrame({
        "omega[eV]": -eraw*KT,
        "momentum": kraw,
        "fit momentum data": K,
        "calculated imaginary part of self energy[eV]": IMS*KT,
        "calculated photoemission peak width(FWHM)": FWHM
    })
    if config["enables"]["enb_saves"]["enb_all_saves"] and config["enables"]["enb_saves"]["enb_save_logs"]:
        dispersion_fit.to_csv(os.path.join(output_folder, "dispersion_fit.csv"), index=False)
        dispersion_data_output.log_to_json(os.path.join(output_folder, "log.json"))


if __name__ == "__main__":
    main()
