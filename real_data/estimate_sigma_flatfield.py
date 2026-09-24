"""
Standalone check/reference run of data_utils.estimate_sigma_I +
propagate_sigma_montecarlo (see that module for the method). Not imported
by the other scripts -- they call data_utils directly -- this is just for
inspecting the numbers on their own.

Usage: python estimate_sigma_flatfield.py
"""
from __future__ import annotations

import data_utils as du


def main():
    data = du.load_real_intensities()
    I_meas, I_ref = data["I_meas"], data["I_ref"]

    sigma_I = du.estimate_sigma_I(I_ref)
    print(f"\nsigma_I (from I_ref flat-field, DC+1st-harmonic residual): {sigma_I:.6e}")

    print("\nRunning Monte Carlo noise propagation (50 trials)...")
    result = du.propagate_sigma_montecarlo(I_meas, I_ref, sigma_I, n_trials=50, seed=0)
    print("\n=== Flat-field-based sigma ===")
    for k, v in result.items():
        print(f"  {k} = {v:.6e}")


if __name__ == "__main__":
    main()
