# imports
import os
from pyEliashMEM.utils.format_data_output import DispersionData
from pyEliashMEM.utils.read_inputs import read_and_prepare_data
from pyEliashMEM.plots.plot_momentum_energy_curve import plot_momentum_energy_curve, plot_momentum_energy_curve_with_fit
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
    CHI0, S, Q, D1, IMS, EBX, EBY, EBDX, EBDY, LAMBDA, DLAMBDA, OMEGALOG, dispersion_data_output = \
        score_output(params, KERN, D, SIGMA, A, M, ALPHA, ND, Y, Y1, DY1, OMEGABIN, EM, dispersion_data_output)
    eraw, KERN, D1, K, IMS = dispersion_output(params, KT, eraw, Y1, DY1, A, A1, A2)
    dispersion_data_output.log_to_json(os.path.join(output_folder, "log.json"))
    pass


if __name__ == "__main__":
    main()
