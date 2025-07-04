import os
import yaml
from pyEliashMEM.utils.read_dispersion_in_file import read_and_shift_dispersion_data

def read_inputs(filename: str = "pyEliashMEM\inputs.yaml"):

    with open(filename, 'r') as f:
        config = yaml.safe_load(f)

    return config["inputs"]


def read_parameters_in_file(filepath):
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


def read_and_prepare_data():
    inputs = read_inputs()
    filepath_ini = os.path.join(inputs["input_parameters_folder"], inputs["input_parameters_file"])
    params = read_parameters_in_file(filepath_ini)
    filepath_dispersion = os.path.join(inputs["input_parameters_folder"], params["DATAIN"])
    eraw, kraw = read_and_shift_dispersion_data(filepath_dispersion, params)
    return inputs, params, eraw, kraw