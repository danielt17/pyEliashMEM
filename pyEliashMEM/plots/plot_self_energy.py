import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import os

plt.style.use(['science', 'ieee'])
plt.rcParams.update({
    "figure.dpi": 300,  # High DPI for clarity
    "figure.figsize": (5, 3.5),
    "font.size": 12,    # Font size for readability
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 1,
    "lines.markersize": 6,
    "font.family": "serif",
    "font.serif": ["CMU Serif"],
})
plt.rcParams['text.usetex'] = True


def plot_self_energy_real_part(self_energy: pd.DataFrame, config: dict, output_folder: str) -> None:
    """
        Plots the momentum-energy curve from raw dispersion data and saves/displays it
        based on the configuration flags.

        The plot shows energy (shifted by EF) versus momentum (shifted by KF),
        styled with blue circle markers and no connecting lines.

        Parameters:
            eraw (np.ndarray): Array of energy values (shifted by EF).
            kraw (np.ndarray): Array of momentum values (shifted by KF).
            params (dict): Dictionary of simulation parameters containing at least:
                - 'EF' (float): Fermi energy in eV.
                - 'KF' (float): Fermi momentum.
            config (dict): Configuration dictionary controlling plot enabling and saving.
                Expected keys:
                  - enables -> enb_plots -> enb_all_plots (bool)
                  - enables -> enb_plots -> enb_momentum_energy_curve (bool)
                  - enables -> enb_saves -> enb_all_saves (bool)
                  - enables -> enb_saves -> enb_save_figures (bool)
                  - enables -> enb_plots -> enb_show_plots (bool)
            output_folder (str): Directory path where plot files are saved.

        Returns:
            None

        Raises:
            KeyError: If expected keys are missing from the `config` or `params`.
            OSError: If saving the plot fails due to invalid output folder.
    """
    if config["enables"]["enb_plots"]["enb_all_plots"] and config["enables"]["enb_plots"]["enb_eliashberg_function"]:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.errorbar(-self_energy["omega[meV]"],
                    self_energy["real part of self energy[meV]"],
                    yerr=self_energy["error bars[meV]"],
                    fmt="--",  # Dashed line for fit
                    color='black',
                    ecolor='black',  # Color for error bars
                    elinewidth=1,  # Error bar line width
                    capsize=3,  # Little horizontal lines at error bar ends
                    label="Initial guess of real part of self-energy")
        ax.plot(-self_energy["omega[meV]"],
                self_energy["fit real part of self energy[meV]"],
                marker="",
                linestyle="-",
                markerfacecolor='none',
                markeredgecolor='blue',
                markeredgewidth=1.5,
                color='blue',
                label="Fit of real part of self-energy")
        ax.set_title(
            rf"Estimated real part of self-energy $Re\mathrm{{\Sigma}} (\mathrm{{\epsilon}})$")
        ax.set_xlabel(r"$\mathrm{{\epsilon}} - \mathrm{{\epsilon}}_{{F}} [meV]$")
        ax.set_ylabel(r"$Re\mathrm{{\Sigma}} (\mathrm{{\epsilon}}) [meV]$")
        ax.legend()
        ax.grid(True)
        if config["enables"]["enb_saves"]["enb_all_saves"] and config["enables"]["enb_saves"]["enb_save_figures"]:
            file_name = "self_energy_real_part.svg"
            plt.savefig(os.path.join(output_folder, file_name), dpi=300, bbox_inches='tight')
        if config["enables"]["enb_plots"]["enb_show_plots"]:
            plt.show()