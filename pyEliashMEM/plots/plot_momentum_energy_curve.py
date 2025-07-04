import matplotlib.pyplot as plt
import numpy as np
import scienceplots


# plt.style.use(['science', 'nature'])
plt.style.use(['science', 'no-latex'])
plt.rcParams.update({
    "figure.dpi": 300,  # High DPI for clarity
    "figure.figsize": (6, 3.5),
    "font.size": 14,    # Font size for readability
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 1,
    "lines.markersize": 6,
    "font.family": "serif",
    "font.serif": ["CMU Serif"],
})


def plot_momentum_energy_curve(eraw: np.array, kraw: np.array, params: dict, config: dict):
    if config["enables"]["enb_plots"]["enb_all_plots"] and config["enables"]["enb_plots"]["enb_momentum_energy_curve"]:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(kraw, eraw)
        ax.set_title(f"Momentum energy curve, EF = {params['EF']} [eV], KF = {params['KF']} [A^-1]")
        ax.set_xlabel(f"k-KF [A^-1]")
        ax.set_ylabel(f"E-EF [eV]")
        if config["enables"]["enb_saves"]["enb_all_saves"] and config["enables"]["enb_saves"]["enb_save_figures"]:
            plt.savefig("")
    pass