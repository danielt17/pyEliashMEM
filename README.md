# pyEliashMEM

**pyEliashMEM** is a Python-based implementation inspired by the original [EliashMEM program](https://web2.ph.utexas.edu/~jrshi/MEM.html), designed to extract the **Eliashberg spectral function** from photoemission data obtained in **Angle-Resolved Photoemission Spectroscopy (ARPES)** experiments.

The Eliashberg function characterizes the electron-boson coupling, typically the **electron-phonon interaction**, which plays a central role in superconductivity and many-body physics.

This tool performs a deconvolution of high-resolution ARPES spectra using the **Maximum Entropy Method (MEM)** to overcome the numerical instability associated with direct inversion methods.

The goal of pyEliashMEM is to streamline and simplify this analysis, making it more accessible to the research community.

> **Note**: If you use this software, please **cite the original work** by Junren Shi *et al.* [2] and this GitHub page.

---

## Features

- Input: High-resolution ARPES photoemission data.
- Output: Eliashberg spectral function α²F(ω).
- Method: Maximum Entropy Method (MEM) for stable inversion.
- Written in Python for flexibility and integration.

---

## References

1. [EliashMEM program by Junren Shi](https://web2.ph.utexas.edu/~jrshi/MEM.html)  
2. Junren Shi, *et al.*, [**Direct Extraction of the Eliashberg Function for Electron-Phonon Coupling: A Case Study of Be(10̄10)**](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.92.186401), *Physical Review Letters*, **92**, 186401 (2004).  
3. M. Jarrell and J.E. Gubernatis, [**Bayesian inference and the analytic continuation of imaginary-time quantum Monte Carlo data**](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.44.6011), *Physical Review B*, **44**, 6011 (1991).

---

## License

This project is licensed under the [MIT License](LICENSE).
