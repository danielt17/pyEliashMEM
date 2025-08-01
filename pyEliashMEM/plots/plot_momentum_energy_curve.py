import matplotlib.pyplot as plt
import numpy as np
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
plt.rcParams['text.usetex'] = False

def plot_momentum_energy_curve(eraw: np.array, kraw: np.array, params: dict, config: dict, output_folder: str) -> None:
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
    if config["enables"]["enb_plots"]["enb_all_plots"] and config["enables"]["enb_plots"]["enb_momentum_energy_curve"]:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(kraw,
                eraw,
                marker="o",
                linestyle="",
                markerfacecolor='none',
                markeredgecolor='blue',
                markeredgewidth=1.5,
                color='blue')
        ax.set_title(
            rf"Momentum-energy curve, "
            rf"$\mathrm{{E}}_F = {params['EF'] * 1e3}$ [meV], "
            rf"$\mathrm{{k}}_F = {params['KF']}$ [$\mathrm{{\AA}}^{{-1}}$]"
        )
        ax.set_xlabel(r"$\mathrm{k} - \mathrm{k}_F$ [$\mathrm{\AA}^{-1}$]")
        ax.set_ylabel(r"$\mathrm{E} - \mathrm{E}_F$ [eV]")
        ax.grid(True)
        if config["enables"]["enb_saves"]["enb_all_saves"] and config["enables"]["enb_saves"]["enb_save_figures"]:
            file_name = "momentum_energy_curve.svg"
            plt.savefig(os.path.join(output_folder, file_name), dpi=300, bbox_inches='tight')
        if config["enables"]["enb_plots"]["enb_show_plots"]:
            plt.show()


def plot_momentum_energy_curve_with_fit(eraw: np.array, kraw: np.array, predicted_curve: np.array,
                                        params: dict, config: dict, output_folder: str) -> None:
    """
    Plots the momentum-energy curve from raw dispersion data along with a fitted estimation curve,
    then saves and/or displays the plot according to configuration flags.

    The plot shows energy (shifted by EF) versus momentum (shifted by KF),
    with experimental data as blue circle markers (no connecting lines) and
    the fitted curve as a red dashed line.

    Parameters:
        eraw (np.ndarray): Array of energy values (shifted by EF).
        kraw (np.ndarray): Array of momentum values (shifted by KF).
        predicted_curve (np.ndarray): Array of fitted energy values corresponding to `kraw`.
        params (dict): Dictionary of simulation parameters containing at least:
            - 'EF' (float): Fermi energy in eV.
            - 'KF' (float): Fermi momentum.
        config (dict): Configuration dictionary controlling plot enabling and saving.
            Expected keys:
              - enables -> enb_plots -> enb_all_plots (bool)
              - enables -> enb_plots -> enb_momentum_energy_curve_with_fit (bool)
              - enables -> enb_saves -> enb_all_saves (bool)
              - enables -> enb_saves -> enb_save_figures (bool)
              - enables -> enb_plots -> enb_show_plots (bool)
        output_folder (str): Directory path where plot files are saved.

    Returns:
        None

    Raises:
        KeyError: If expected keys are missing from the `config` or `params`.
        OSError: If saving the plot fails due to an invalid output folder.
    """
    if config["enables"]["enb_plots"]["enb_all_plots"] and config["enables"]["enb_plots"]["enb_momentum_energy_curve_with_fit"]:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(kraw,
                eraw,
                marker="o",
                linestyle="",
                markerfacecolor='none',
                markeredgecolor='blue',
                markeredgewidth=1.5,
                color='blue',
                label="Experiment")
        ax.plot(kraw,
                predicted_curve,
                marker="",
                linestyle="--",
                markerfacecolor='red',
                markeredgecolor='red',
                markeredgewidth=1.5,
                color='red',
                label="Fit")
        ax.set_title(
            rf"Momentum-energy curve with fit, "
            rf"$\mathrm{{E}}_F = {params['EF'] * 1e3}$ [meV], "
            rf"$\mathrm{{k}}_F = {params['KF']}$ [$\mathrm{{\AA}}^{{-1}}$]"
        )
        ax.set_xlabel(r"$\mathrm{k} - \mathrm{k}_F$ [$\mathrm{\AA}^{-1}$]")
        ax.set_ylabel(r"$\mathrm{E} - \mathrm{E}_F$ [eV]")
        ax.legend()
        ax.grid(True)
        if config["enables"]["enb_saves"]["enb_all_saves"] and config["enables"]["enb_saves"]["enb_save_figures"]:
            if config["enables"]["enb_estimations"]["enb_all_estimations"] and \
                    config["enables"]["enb_estimations"]["enb_estimation_momentum_energy_curve"]:
                file_name = "momentum_energy_curve_with_my_fit.svg"
            else:
                file_name = "momentum_energy_curve_with_paper_fit.svg"
            plt.savefig(os.path.join(output_folder, file_name), dpi=300, bbox_inches='tight')
        if config["enables"]["enb_plots"]["enb_show_plots"]:
            plt.show()
