# Bayesian Joint Reconstruction and Uncertainty Quantification for Limited-Data Multi-Contrast X-ray CT

Code accompanying *Bayesian Joint Reconstruction and Uncertainty Quantification
for Limited-Data Multi-Contrast X-ray CT*, Hassan Yazdanian and Henrik Mäkinen.

Talbot–Lau interferometry measures three contrasts at once — attenuation,
differential phase and dark-field. This repository reconstructs the
corresponding fields (μ, δ, ε) jointly from the retrieved sinograms under a
single posterior, with a joint total-variation prior coupling the channels, and
quantifies the reconstruction uncertainty by NUTS sampling. Both a MAP estimate
and posterior mean/standard-deviation maps are produced for four acquisition
conditions: full sampling, reduced angles, reduced phase steps, and both.

## Installation

```bash
conda env create -f environment.yml
conda activate bayes-mc-ct
```

ASTRA's CUDA build is required for the FBP baseline. Everything else runs on
CPU, but posterior sampling is impractically slow without a GPU (see
[Runtimes](#runtimes)).

## Layout

```
common/            Physics, shared by both studies
  TLI_2D_forward.py    Talbot-Lau forward model and fan-beam projector
  phase_stepping.py    FFT-based phase-stepping retrieval (T, DPC, D, P)
  fanbeam_astra.py     ASTRA fan-beam FBP baseline
  scenario_utils.py    Angle/step subsetting, noise injection, support mask

Synthetic_data/    Phantom study (ground truth available)
  phantoms.py, create_data.py    Phantom definition and data generation
  recon/                         MAP reconstruction, parameter tuning, figures
  uq/                            NUTS sampling, calibration, figures

real_data/         Experimental Talbot-Lau study
  data_utils.py        Loading, retrieval, geometry, noise estimation
  map_real.py          MAP reconstruction (TV and joint-TV)
  nuts_real.py         Posterior sampling
  run_scenarios.py     FBP / TV / JTV across the four scenarios
  cv_alpha_real*.py    Held-out-angle cross-validation for the prior weights
  uq_summary.py        Per-scenario ESS, R-hat and posterior width tables

forward_validate/  Round-trip check of the forward model against measurement
```

Each study writes to its own `obs/` directory, which is not tracked.

## Data

`real_data/obs/` holds the experimental measurements: a 360° scan of 361
projection angles × 10 phase steps × 843 detector pixels, as `I_meas_central.npy`
(sample) and `I_ref_central.npy` (flat field), with `angles.npy` and `steps.npy`.
These are the only inputs that cannot be regenerated. Synthetic data is built by
`Synthetic_data/create_data.py`.

## Reproducing the results

**Synthetic study.** Data are simulated at 512² with a bilinear projector and
reconstructed at 256² with a nearest-neighbour projector, so the reconstruction
never inverts the operator that generated the data.

```bash
python Synthetic_data/create_data.py
python Synthetic_data/recon/sweep_tv_beta_synthetic.py --stage alpha_beta
python Synthetic_data/recon/tune_alpha.py
python Synthetic_data/recon/run_scenarios.py
python Synthetic_data/uq/nuts_synthetic.py --scenario full --alpha_joint 29.55
python Synthetic_data/uq/uq_calibration.py
```

**Experimental study.**

```bash
python real_data/cv_alpha_real.py          # per-channel prior weights
python real_data/cv_alpha_real_joint.py    # joint-prior weight
python real_data/run_scenarios.py          # FBP / TV / JTV, four scenarios
python real_data/nuts_real.py --scenario full --alpha_joint 0    # TV
python real_data/nuts_real.py --scenario full                    # joint TV
python real_data/uq_summary.py
```

`--scenario` takes `full`, `sparse_angle`, `sparse_step` or `combined`.
Every regularization parameter can be overridden on the command line
(`--alpha_mu`, `--alpha_delta`, `--alpha_eps`, `--alpha_joint`, `--tv_beta`,
`--lambda0`), so an operating point can be changed without editing the source.

**Forward-model validation.**

```bash
python forward_validate/run_validate_TLI_forward_vs_real.py
```

## Noise model

The likelihood is defined on the retrieved sinograms, so its σ must match the
acquisition being modelled. σ is estimated by Monte Carlo propagation of the
flat-field intensity noise through the actual retrieval
(`data_utils.sigma_for_stepping`), and therefore depends on how many phase steps
a scenario records: retrieving from 5 of the 10 recorded steps averages half as
much noise away and raises σ by √2. Reusing a full-scan σ for the reduced-step
scenarios would make the likelihood twice as confident as the data warrant.

## Runtimes

Measured on a single NVIDIA GPU. MAP reconstruction is seconds to a minute per
scenario. Posterior sampling dominates:

| Study | Scenario | Per chain |
|---|---|---|
| Experimental (512², 361 angles) | `full`, `sparse_step` | ~9.5 h |
| Experimental (512², 30 angles) | `sparse_angle`, `combined` | ~50 min |
| Synthetic (256²) | all eight runs combined | ~6.3 h |

Each run uses 2 chains of 1000 samples after 200 warmup steps. Sample archives
are ~2.3 GB per run and are not tracked; posterior summaries are written
alongside them as `posterior_mean_std.npz`.

## License

MIT — see [LICENSE](LICENSE).
