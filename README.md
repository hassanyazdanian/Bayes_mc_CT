# Bayesian Joint Reconstruction and Uncertainty Quantification for Limited-Data Multi-Contrast X-ray CT

Code accompanying *Bayesian Joint Reconstruction and Uncertainty Quantification
for Limited-Data Multi-Contrast X-ray CT*, Hassan Yazdanian and Henrik Mäkinen.

Talbot–Lau interferometry measures three contrasts at once: attenuation,
differential phase and dark-field. This repository reconstructs the
corresponding fields (μ, δ, ε) jointly from the retrieved sinograms under a
single posterior, with a joint total-variation prior coupling the channels, and
quantifies the reconstruction uncertainty by NUTS sampling. Both a MAP estimate
and posterior mean/standard-deviation maps are produced for four acquisition
conditions: full data, sparse angles, sparse phase steps, and both combined.

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
common/            Physics and shared numerics, used by both studies
  TLI_2D_forward.py    Talbot-Lau forward model and fan-beam projector
  phase_stepping.py    FFT-based phase-stepping retrieval (T, DPC, D, P)
  fanbeam_astra.py     ASTRA fan-beam FBP baseline
  scenario_utils.py    Angle/step subsetting, noise injection, support mask
  map_tv_jtv.py        MAP solver with TV and joint-TV priors
  posterior.py         Sample loading and posterior mean/std reduction

synthetic_data/    Phantom study (ground truth available)
  phantoms.py, create_data.py    Phantom definition and data generation
  recon/               run_scenarios.py, tune_alpha*.py, sweep_tv_beta.py,
                       fig_map_comparison.py
  uq/                  nuts_synthetic.py, post_process_nuts.py,
                       uq_calibration.py, diagnose_convergence.py,
                       fig_uq_comparison.py

real_data/         Experimental Talbot-Lau study
  extract_central_row.py  Builds the obs/ inputs from the raw scan (see Data)
  data_utils.py        Loading, retrieval, geometry, noise estimation
  recon/               map_real.py, run_scenarios.py, cv_alpha_real*.py,
                       sweep_tv_beta_lambda0.py, fig_map_comparison.py
  uq/                  nuts_real.py, post_process_real.py, uq_summary.py,
                       diagnose_convergence.py, fig_uq_comparison.py

forward_validate/  Round-trip check of the forward model against measurement
```

Each study writes its outputs to its own `obs/` directory. These outputs are
not tracked; the experimental inputs in `real_data/obs/` are (see Data).
Scripts locate `common/` and their own study directory relative to their
own path, so they can be run from anywhere.

## Data

`real_data/obs/` holds the experimental measurements used in the paper: the
central detector row of a 360° Talbot–Lau scan, with 361 projection angles ×
10 phase steps × 843 detector pixels, as `I_meas_central.npy` (sample) and
`I_ref_central.npy` (flat field), with `angles.npy` (angle indices 1–361, i.e.
0°–360° in 1° steps) and `steps.npy` (step indices 1–10).

The row comes from the 35 kV water-container scan of the sugar-solution
phantom in H. Mäkinen et al., *Biomed. Phys. Eng. Express* 10, 045045 (2024)
(acquisition folder `pmma_stick_phantom/scan22`), whose raw images are part of
the open dataset (about 463 GB), licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):

> H. Mäkinen, H. Suhonen, T. Siiskonen, C. David, and S. Huotari, "Dataset for
> paper Optimization of contrast and dose in X-ray phase-contrast tomography
> with a Talbot-Lau interferometer," University of Helsinki, 2023.
> [https://doi.org/10.23729/60132ae3-1ce3-41eb-935f-d0721ad464aa](https://doi.org/10.23729/60132ae3-1ce3-41eb-935f-d0721ad464aa)

`real_data/extract_central_row.py` rebuilds the four files from the scan's
`raw/` folder and needs only numpy. It takes the transaxial detector line
through the detector centre (column 768 of the 864 × 1536 raw images), keeps
rows 10–852, and shifts this window at each angle by the random sample offset
recorded in `scan.csv` (−10 to 10 pixels, applied during acquisition to reduce
ring artefacts), for both the object (`im`) and reference (`ref`) images:

```bash
python real_data/extract_central_row.py /path/to/pmma_stick_phantom/scan22/raw
```

Synthetic data is built by `synthetic_data/create_data.py`.

## Reproducing the results

**Synthetic study.** Data are simulated at 512² with a bilinear projector and
reconstructed at 256² with a nearest-neighbour projector, so the reconstruction
never inverts the operator that generated the data.

```bash
python synthetic_data/create_data.py
python synthetic_data/recon/sweep_tv_beta.py --stage alpha_beta
python synthetic_data/recon/tune_alpha.py
python synthetic_data/recon/run_scenarios.py
python synthetic_data/uq/nuts_synthetic.py --phantom multicontrast --scenario full \
    --alpha_joint 0 --num_samples 1000 --warmup_steps 500        # TV
python synthetic_data/uq/nuts_synthetic.py --phantom multicontrast --scenario full \
    --alpha_joint 29.55 --num_samples 1000 --warmup_steps 500    # joint TV
python synthetic_data/uq/uq_calibration.py
```

**Experimental study.**

```bash
python real_data/recon/cv_alpha_real.py          # per-channel prior weights
python real_data/recon/cv_alpha_real_joint.py    # joint-prior weight
python real_data/recon/run_scenarios.py          # FBP / TV / JTV, four scenarios
python real_data/uq/nuts_real.py --scenario full --alpha_joint 0 --num_samples 1000    # TV
python real_data/uq/nuts_real.py --scenario full --num_samples 1000                    # joint TV
python real_data/uq/uq_summary.py
```

`--scenario` takes `full`, `sparse_angle`, `sparse_step` or `combined`. The
sampling scripts default to 200 samples and 200 warmup iterations for quick
runs; the commands above use the settings of the paper (see
[Runtimes](#runtimes)).
Every regularization parameter can be overridden on the command line
(`--alpha_mu`, `--alpha_delta`, `--alpha_eps`, `--alpha_joint`, `--tv_beta`,
`--lambda0`), so an operating point can be changed without editing the source.

**Forward-model validation.**

```bash
python forward_validate/run_validate.py
```

## Noise model

The likelihood is defined on the retrieved sinograms, so its σ must match the
acquisition being modelled. σ is estimated by Monte Carlo propagation of the
flat-field intensity noise through the actual retrieval
(`data_utils.sigma_for_stepping`), and therefore depends on how many phase steps
a scenario records: retrieving from 5 of the 10 recorded steps averages half as
much noise away and raises σ by √2. Reusing a full-scan σ for the sparse-step
scenarios would make the likelihood twice as confident as the data warrant.

## Runtimes

Measured on an NVIDIA RTX 6000 Ada Generation GPU (48 GB). MAP reconstruction
takes seconds to a minute per scenario. Posterior sampling dominates:

| Study | Scenarios | Per chain |
|---|---|---|
| Experimental (512², 361 angles) | `full`, `sparse_step` | ~9.5 h |
| Experimental (512², 30 angles) | `sparse_angle`, `combined` | ~50 min |
| Synthetic (256², 180 angles) | `full`, `sparse_step` | ~47 min |
| Synthetic (256², 30 angles) | `sparse_angle`, `combined` | ~9.5 min |

Each run draws 2 chains of 1000 samples, after 500 warmup iterations in the
synthetic study and 200 in the experimental study. Sample archives are ~2.3 GB
per run and are not tracked; posterior summaries are written alongside them as
`posterior_mean_std.npz`.

## License

MIT (see [LICENSE](LICENSE)).
