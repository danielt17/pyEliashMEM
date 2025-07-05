from dataclasses import dataclass
import scipy.constants


@dataclass(frozen=True)
class Constants:
    NMAX: int = 1000
    PI: float = scipy.constants.pi
    PI2: float = scipy.constants.pi**2  # PI squared
    ZERO: float = 1e-20
    EPS: float = 1e-4

