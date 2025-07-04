# imports
import os
from pyEliashMEM.utils.read_inputs import read_inputs, read_parameters_in_file
from pyEliashMEM.utils.read_dispersion_in_file import read_and_shift_dispersion_data


def main():
    inputs = read_inputs()
    filepath_ini = os.path.join(inputs["input_parameters_folder"], inputs["input_parameters_file"])
    params = read_parameters_in_file(filepath_ini)
    filepath_dispersion = os.path.join(inputs["input_parameters_folder"], params["DATAIN"])
    eraw, kraw = read_and_shift_dispersion_data(filepath_dispersion, params)
    pass


if __name__ == "__main__":
    main()
