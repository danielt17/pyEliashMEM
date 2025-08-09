from dataclasses import dataclass, field, asdict
from typing import Optional, List
import json


@dataclass
class DispersionData:
    """
    Holds all input, configuration, and results parameters for a dispersion analysis.

    The class:
      - Stores input file paths, model identifiers, and numerical configuration values.
      - Keeps track of both raw configuration parameters (e.g., ECUTOFF, A1, A2)
        and post-analysis results (e.g., ND, CHI², ALPHA).
      - Accommodates both scalar parameters and list-based bin definitions.
      - Separates input/configuration fields from results fields clearly.

    Parameters (Input / Configuration):
        input_data (Optional[str]): Path to the input data file (e.g., 'dispersion.txt').
        input_model (Optional[str]): Model type identifier (e.g., 'NONE').
        output_file_prefix (Optional[str]): Prefix for generated output files.
        total_data_points (Optional[int]): Number of data points in the input dataset.
        total_omega_points (Optional[int]): Number of omega points for analysis.
        cutoff_energy (Optional[float]): Energy cutoff value (ECUTOFF) for filtering.
        temperature (Optional[float]): System temperature in Kelvin.
        a1 (Optional[float]): Linear dispersion coefficient from input.
        a2 (Optional[float]): Quadratic dispersion coefficient from input.
        ef (Optional[float]): Fermi energy.
        kf (Optional[float]): Fermi momentum.
        data_error_bar (Optional[str]): Data error handling method ('AUTOMATIC' or manual).
        data_error_bar_value (Optional[float]): Numeric value of the error bar if manual.
        data_error_bar_slop (Optional[float]): Slope parameter for error bar scaling.
        mem_method (Optional[str]): Maximum Entropy Method type (e.g., 'CLASSIC').
        target_chi_squared (Optional[float]): Target χ² value to reach in fitting.
        max_alpha (Optional[float]): Maximum value for α parameter in fitting.
        alpha_step (Optional[float]): Step size for α parameter scanning.
        max_iteration_num (Optional[int]): Maximum number of fitting iterations.
        def_model_omega_d (Optional[float]): Default model Debye frequency ω_D.
        def_model_omega_m (Optional[float]): Default model maximum frequency ω_M.
        def_model_lambda0 (Optional[float]): Default initial coupling constant λ₀.
        weight_exponent (Optional[float]): Exponent used in weighting function during fitting.
        total_out_bins (Optional[int]): Number of output frequency bins.
        out_bin_values (Optional[List[float]]): List of bin boundary values for output bins.

    Parameters (Results):
        nd (Optional[int]): Number of discrete elements or bins after calculation.
        sigma_bar (Optional[float]): Average sigma (σ̄) value from the analysis.
        a1_est (Optional[float]): Estimated linear dispersion coefficient after fitting.
        a2_est (Optional[float]): Estimated quadratic dispersion coefficient after fitting.
        chi2 (Optional[float]): Final chi-squared (χ²) fitting metric.
        q (Optional[float]): Goodness-of-fit Q-value (probability).
        alpha (Optional[float]): Best-fit α parameter.
        dalpha (Optional[float]): Uncertainty in α (Δα).
        lambda_ (Optional[float]): Estimated electron-phonon coupling constant λ.
        d_lambda (Optional[float]): Uncertainty or error in λ (Δλ).
        omega_log (Optional[float]): Logarithmic average phonon frequency ω_log.

    Notes:
      - This dataclass is intended as a container only; no calculations are
        performed internally.
      - All values are initialized to None and must be set explicitly after instantiation.
      - Field names closely match those found in the text-format output files for easy parsing.
      - The `init=False` in fields indicates values are typically assigned post-creation.
    """
    input_data: Optional[str] = field(default=None, init=False)
    input_model: Optional[str] = field(default=None, init=False)
    output_file_prefix: Optional[str] = field(default=None, init=False)
    total_data_points: Optional[int] = field(default=None, init=False)
    total_omega_points: Optional[int] = field(default=None, init=False)
    cutoff_energy: Optional[float] = field(default=None, init=False)
    temperature: Optional[float] = field(default=None, init=False)
    a1: Optional[float] = field(default=None, init=False)
    a2: Optional[float] = field(default=None, init=False)
    ef: Optional[float] = field(default=None, init=False)
    kf: Optional[float] = field(default=None, init=False)
    data_error_bar: Optional[str] = field(default=None, init=False)
    data_error_bar_value: Optional[float] = field(default=None, init=False)
    data_error_bar_slop: Optional[float] = field(default=None, init=False)
    mem_method: Optional[str] = field(default=None, init=False)
    target_chi_squared: Optional[float] = field(default=None, init=False)
    max_alpha: Optional[float] = field(default=None, init=False)
    alpha_step: Optional[float] = field(default=None, init=False)
    max_iteration_num: Optional[int] = field(default=None, init=False)
    def_model_omega_d: Optional[float] = field(default=None, init=False)
    def_model_omega_m: Optional[float] = field(default=None, init=False)
    def_model_lambda0: Optional[float] = field(default=None, init=False)
    weight_exponent: Optional[float] = field(default=None, init=False)
    total_out_bins: Optional[int] = field(default=None, init=False)
    out_bin_values: Optional[List[float]] = field(default=None, init=False)

    # Results section
    nd: Optional[int] = field(default=None, init=False)
    sigma_bar: Optional[float] = field(default=None, init=False)
    a1_est: Optional[float] = field(default=None, init=False)
    a2_est: Optional[float] = field(default=None, init=False)
    chi2: Optional[float] = field(default=None, init=False)
    q: Optional[float] = field(default=None, init=False)
    alpha: Optional[float] = field(default=None, init=False)
    dalpha: Optional[float] = field(default=None, init=False)
    lambda_: Optional[float] = field(default=None, init=False)
    d_lambda: Optional[float] = field(default=None, init=False)
    omega_log: Optional[float] = field(default=None, init=False)

    def log_to_json(self, output_path: str):
        """
        Save the current DispersionData instance as a JSON file.

        Args:
            output_path (str): Path to the output JSON file.

        This method serializes all fields of the dataclass, including configuration
        and results, into JSON format for easy external use, storage, or debugging.
        """
        log = asdict(self)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=4)

