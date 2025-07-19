# imports
import os
import numpy as np
from pyEliashMEM.utils.read_inputs import read_and_prepare_data
from pyEliashMEM.plots.plot_momentum_energy_curve import plot_momentum_energy_curve, plot_momentum_energy_curve_with_fit
from pyEliashMEM.estimation.fit_predict_momentum_energy_curve import fit_predict_momentum_energy_curve, estimate_error
from pyEliashMEM.estimation.constraints import model_constraint
from pyEliashMEM.estimation.utils import setup_kernel, IMSIGMA, weight, intavg
from pyEliashMEM.maximum_entropy_method.mem_main import iterative_mem_fit
from pyEliashMEM.maximum_entropy_method.mem_utils import chi, calc_score


def main():
    config, params, eraw, kraw, output_folder = read_and_prepare_data()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plot_momentum_energy_curve(eraw, kraw, params, config, output_folder)
    predicted_curve, ND, Y, D, K = fit_predict_momentum_energy_curve(eraw, kraw, params, config)
    plot_momentum_energy_curve_with_fit(eraw, kraw, predicted_curve, params, config, output_folder)
    SIGMA, Y = estimate_error(ND, Y, D, params)
    KT, Y, D, SIGMA, OMEGABIN, OMEGAD, OMEGAM, Y1, M, DY1, inv_sigma2, Ksq, K4, K3, X1, X2, X12, XX = \
        model_constraint(params["KT"], Y, K, D, SIGMA, params["OMEGABIN"], params["NBIN"], params["OMEGAD"],
                     params["OMEGAM"], params["MODEL"], params["NA"], params["LAMBDA0"], ND)
    KERN = setup_kernel(ND, params["NA"], Y, Y1, DY1)

    A1, A2, D, J, A, DA, KERN, SIGMA, ALPHA, DALPHA, EM = iterative_mem_fit(params["A1"], params["A2"], ND,
                                                                            params["NA"], params["ITERNUM"],
                                                                            params["METHOD"], params["FITBPD"], KERN, D,
                                                                            SIGMA, M, params["ALPHA"], params["DALPHA"],
                                                                            params["XCHI"], Y, K, KT, X1, X2, X12)
    CHI0 = chi(KERN, D, SIGMA, A)
    S = calc_score(A, M)
    Q = CHI0/ 2 - ALPHA * S
    D1 = KERN @ A
    IMS = np.empty(ND)
    for i in range(ND):
        IMS[i] = IMSIGMA(params["NA"], Y[i], A, Y1, DY1)
    EBX, EBY, EBDX, EBDY = weight(params["NA"], params["NBIN"], OMEGABIN, params["BETA"], A, Y1, DY1, EM)
    LAMBDA, DLAMBDA, OMEGALOG = intavg(A, Y1, DY1, EM)
    eraw *= -1.0 / KT
    KERN = setup_kernel(ND, params["NA"], Y, Y1, DY1)
    D1 = KERN @ A
    K = np.zeros(params["NDRAW"])
    # ND shouldnt be here but ndraw
    # for i in range(params["NDRAW"]):
    for i in range(ND):
        E0 = - (eraw[i] + D1[i]) * KT
        denominator = np.abs(A1) + np.sqrt(A1 ** 2 + 4.0 * A2 * E0)
        K[i] = 2.0 * E0 / denominator * np.sign(-A1)
        IMS = IMSIGMA(params["NA"], eraw[i], A, Y1, DY1)
    pass


if __name__ == "__main__":
    main()
