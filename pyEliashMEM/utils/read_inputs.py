import os
import yaml
from pyEliashMEM.utils.read_dispersion_in_file import read_and_shift_dispersion_data
import numpy as np
from typing import Tuple


def read_inputs(filename: str = "pyEliashMEM\inputs.yaml") -> dict:
    """
    Reads a YAML configuration file and returns the full configuration as a dictionary.

    The YAML file is expected to have the following structure:

        inputs:
          input_parameters_folder: pyEliashMEM\\examples\\Be1010
          input_parameters_file: CONF3.INI
        enables:
          enb_plots:
            enb_all_plots: true
            enb_momentum_energy_curve: true
          enb_estimation: true

    Parameters:
        filename (str): Path to the YAML configuration file.
                        Defaults to "pyEliashMEM\\inputs.yaml".

    Returns:
        dict: A dictionary containing all configuration sections from the YAML file.
              Example top-level keys:
                - 'inputs' (dict): Paths to input data and config files.
                - 'enables' (dict): Feature enable/disable flags.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the YAML file is invalid or malformed.
    """
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)

    return config


def read_parameters_in_file(filepath: str) -> dict:
    """
        Reads a Fortran-style parameter file and extracts simulation parameters.

        The file is expected to contain one parameter per line, with optional inline comments.
        Only the first token on each line is read. The structure and ordering must match
        the expected Fortran input format.

        Parameters:
            filepath (str): Path to the parameter file (e.g., 'CONF3.INI').

        Returns:
            dict: A dictionary containing all extracted parameters, including:
                - File identifiers: 'DATAIN', 'MODEL', 'OUTPRX'
                - Simulation values: 'NDRAW', 'NA', 'ECUTOFF', 'KT', etc.
                - A list of bin edges: 'OMEGABIN' (length NBIN + 1)

        Raises:
            FileNotFoundError: If the specified file cannot be found.
            StopIteration: If the file has fewer lines than expected.
            ValueError: If any value cannot be converted to the expected type.
    """
    with open(filepath, 'r') as f:
        lines = [line.strip().split()[0] for line in f if line.strip()]  # read first word of each line

    # Iterator over lines
    it = iter(lines)

    parameters_in = {
        # File and model identifiers
        "DATAIN": next(it),
        "MODEL": next(it),
        "OUTPRX": next(it),

        # Data parameters
        "NDRAW": int(next(it)),
        "NA": int(next(it)),
        "ECUTOFF": float(next(it)),
        "KT": float(next(it)),

        # Raw data correction
        "FITBPD": int(next(it)),
        "A1": float(next(it)),
        "A2": float(next(it)),
        "EF": float(next(it)),
        "KF": float(next(it)),

        # Error bars
        "ERRB0": float(next(it)),
        "ERRB1": float(next(it)),

        # Fitting parameters
        "METHOD": int(next(it)),
        "ITERNUM": int(next(it)),
        "ALPHA": float(next(it)),
        "DALPHA": float(next(it)),
        "XCHI": float(next(it)),

        # Default model parameters
        "OMEGAD": float(next(it)),
        "OMEGAM": float(next(it)),
        "LAMBDA0": float(next(it)),

        # Beta and binning
        "BETA": float(next(it)),
        "NBIN": int(next(it)),
    }

    # Read OMEGABINs
    nbin = parameters_in["NBIN"]
    parameters_in["OMEGABIN"] = [float(next(it)) for _ in range(nbin + 1)]

    return parameters_in


def read_and_prepare_data() -> Tuple[dict, dict, np.array, np.array, str]:
    """
        Reads input YAML, Fortran-style parameter file, and raw dispersion data.

        This function performs the full initialization step of the simulation:
        1. Reads general inputs from a YAML configuration file.
        2. Loads simulation parameters from a Fortran-style INI file.
        3. Loads and shifts raw energy and momentum data using EF and KF.

        Returns:
            tuple:
                - config (dict): Dictionary with general inputs from `input.yaml`,
                  typically containing:
                    - 'input_parameters_folder' (str)
                    - 'input_parameters_file' (str)
                - params (dict): Dictionary of simulation parameters from the INI file,
                  including EF, KF, NDRAW, and OMEGABIN, among others.
                - eraw (np.ndarray): Energy values (shifted by EF), shape (NDRAW,)
                - kraw (np.ndarray): Momentum values (shifted by KF), shape (NDRAW,)
                - output_folder (str): Path for output data

        Raises:
            FileNotFoundError: If any of the required files are missing.
            KeyError: If required keys are missing in the YAML or INI file.
            ValueError: If numerical conversion fails during parsing.
    """
    config = read_inputs()
    filepath_ini = os.path.join(config["inputs"]["input_parameters_folder"], config["inputs"]["input_parameters_file"])
    params = read_parameters_in_file(filepath_ini)
    filepath_dispersion = os.path.join(config["inputs"]["input_parameters_folder"], params["DATAIN"])
    eraw, kraw = read_and_shift_dispersion_data(filepath_dispersion, params)
    output_folder = os.path.join(config["inputs"]["input_parameters_folder"],params["OUTPRX"])
    return config, params, eraw, kraw, output_folder