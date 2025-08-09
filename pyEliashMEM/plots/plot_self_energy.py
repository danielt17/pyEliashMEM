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


def plot_self_energy_imaginary_part(self_energy: pd.DataFrame, config: dict, output_folder: str) -> None:
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
        ax.plot(-self_energy["omega[meV]"],
                self_energy["calculated imaginary part of self energy[meV]"],
                marker="",
                linestyle="-",
                markerfacecolor='none',
                markeredgecolor='blue',
                markeredgewidth=1.5,
                color='blue')
        ax.set_title(
            rf"Estimated imaginary part of self-energy $Im\mathrm{{\Sigma}} (\mathrm{{\epsilon}})$")
        ax.set_xlabel(r"$\mathrm{{\epsilon}} - \mathrm{{\epsilon}}_{{F}} [meV]$")
        ax.set_ylabel(r"$Im\mathrm{{\Sigma}} (\mathrm{{\epsilon}}) [meV]$")
        ax.grid(True)
        if config["enables"]["enb_saves"]["enb_all_saves"] and config["enables"]["enb_saves"]["enb_save_figures"]:
            file_name = "self_energy_imaginary_part.svg"
            plt.savefig(os.path.join(output_folder, file_name), dpi=300, bbox_inches='tight')
        if config["enables"]["enb_plots"]["enb_show_plots"]:
            plt.show()


def plot_real_part_self_energy_and_eliashberg_function(self_energy: pd.DataFrame,
                                                       eliashberg_function: pd.DataFrame,
                                                       config: dict, output_folder: str) -> None:
    """
    Plots the real part of self-energy on the left y-axis and the Eliashberg spectral
    function on the right y-axis against energy (shifted by EF).
    The right y-axis is scaled from 0 to twice the maximum Eliashberg spectral value.
    """
    if config["enables"]["enb_plots"]["enb_all_plots"] and config["enables"]["enb_plots"]["enb_eliashberg_function"]:

        fig, ax1 = plt.subplots(figsize=(5, 3.5))

        # Left axis: Real part of self-energy
        l1, = ax1.plot(-self_energy["omega[meV]"],
                       self_energy["fit real part of self energy[meV]"],
                       color='blue', linestyle='-', label="Real part of self-energy")
        ax1.set_xlabel(r"$\mathrm{\epsilon} - \mathrm{\epsilon}_F$ [meV]")
        ax1.set_ylabel(r"$Re\,\Sigma(\epsilon)\ [\mathrm{meV}]$", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Right axis: Eliashberg spectral function
        ax2 = ax1.twinx()
        l2, = ax2.plot(-eliashberg_function["omega[meV]"],
                       eliashberg_function["Eliashberg function"],
                       color='black', linestyle='--', label=r"$\alpha^2F(\omega)$")
        ax2.set_ylabel(r"$\alpha^2F(\omega)$ [meV]", color='black')
        ax2.tick_params(axis='y', labelcolor='black')

        # Set y-axis limits for right axis
        max_val = eliashberg_function["Eliashberg function"].max()
        ax2.set_ylim(0, 2 * max_val)

        # Combined title
        ax1.set_title(r"Real Part of Self-Energy and Eliashberg Spectral Function")

        # Grid on left axis
        ax1.grid(True)

        # Merge legends from both axes
        handles, labels = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(handles + handles2, labels + labels2, loc='upper left')

        # Save figure
        if config["enables"]["enb_saves"]["enb_all_saves"] and config["enables"]["enb_saves"]["enb_save_figures"]:
            file_name = "self_energy_real_part_with_eliashberg_function.svg"
            plt.savefig(os.path.join(output_folder, file_name), dpi=300, bbox_inches='tight')

        # Show figure
        if config["enables"]["enb_plots"]["enb_show_plots"]:
            plt.show()


