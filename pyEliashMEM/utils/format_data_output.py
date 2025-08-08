from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DispersionData:
    """
        Holds all input, configuration, and results parameters for a dispersion analysis.

        The class:
          - Stores input file paths, model identifiers, and numerical configuration values.
          - Keeps track of both raw configuration parameters (e.g., ECUTOFF, A1, A2)
            and post-analysis results (e.g., ND, CHI², ALPHA).
          - Accommodates both scalar parameters and list-based bin definitions.

        Parameters:
            input_data (Optional[str]): Path to the input data file (e.g., 'dispersion.txt').
            input_model (Optional[str]): Model type identifier (e.g., 'NONE').
            output_file_prefix (Optional[str]): Prefix for generated output files.
            total_data_points (Optional[int]): Number of data points in the input dataset.
            total_omega_points (Optional[int]): Number of omega points for analysis.
            cutoff_energy (Optional[float]): Energy cutoff value (ECUTOFF) for filtering.
            temperature (Optional[float]): System temperature in Kelvin.
            a1 (Optional[float]): Linear dispersion coefficient.
            a2 (Optional[float]): Quadratic dispersion coefficient.
            ef (Optional[float]): Fermi energy.
            kf (Optional[float]): Fermi momentum.
            data_error_bar (Optional[str]): Data error handling method ('AUTOMATIC' or manual).
            data_error_bar_value (Optional[float]): value of error bar.
            data_error_bar_slop (Optional[float]): value of error bar slop.
            mem_method (Optional[str]): Maximum Entropy Method type (e.g., 'CLASSIC').
            max_iteration_num (Optional[int]): Maximum number of fitting iterations.
            def_model_omega_d (Optional[float]): Default model ω_D value.
            def_model_omega_m (Optional[float]): Default model ω_M value.
            def_model_lambda0 (Optional[float]): Default model λ₀ value.
            weight_exponent (Optional[float]): Weight exponent in fitting algorithm.
            total_out_bins (Optional[int]): Number of output bins.
            out_bin_values (Optional[List[float]]): Numerical values defining bin boundaries.

        Results:
            nd (Optional[int]): Total number of elements post calculation.
            sigma_bar (Optional[float]): σ̄ result from analysis.
            chi2 (Optional[float]): χ² fitting metric.
            q (Optional[float]): Goodness-of-fit Q-value.
            alpha (Optional[float]): Fitted α value.
            lambda_ (Optional[float]): Fitted λ value.
            d_lambda (Optional[float]): Uncertainty Δλ.
            omega_log (Optional[float]): ω_log value from analysis.

        Notes:
            - This dataclass is intended as a container only; no calculations are
              performed internally.
            - All values are initialized to None and must be set after instantiation.
            - The field names match closely to those found in the text-format output
              for easier parsing.
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
    chi2: Optional[float] = field(default=None, init=False)
    q: Optional[float] = field(default=None, init=False)
    alpha: Optional[float] = field(default=None, init=False)
    lambda_: Optional[float] = field(default=None, init=False)
    d_lambda: Optional[float] = field(default=None, init=False)
    omega_log: Optional[float] = field(default=None, init=False)

