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
central detector row of a 360° Talbot–Lau scan of a PMMA rod with sugar–water
cavities, with 361 projection angles × 10 phase steps × 843 detector pixels, as
`I_meas_central.npy` (sample) and `I_ref_central.npy` (flat field), with
`angles.npy` (angle indices 1–361, i.e. 0°–360° in 1° steps) and `steps.npy`
(step indices 1–10).

The raw phase-stepping images of this scan (35 kV, 40 mA, 18 s per step) are
openly available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
as `PC_PMMA_Phantom/raw.zip` (15.5 GB) in

> H. Mäkinen, "Data for paper: Bayesian Joint Reconstruction and Uncertainty
> Quantification for Limited-Data Multi-Contrast X-ray CT," University of
> Helsinki, 2026.
> [https://doi.org/10.23729/fd-769dae34-0245-386f-a856-92e1adc60f18](https://doi.org/10.23729/fd-769dae34-0245-386f-a856-92e1adc60f18)

They are repackaged from the dataset
([https://doi.org/10.23729/60132ae3-1ce3-41eb-935f-d0721ad464aa](https://doi.org/10.23729/60132ae3-1ce3-41eb-935f-d0721ad464aa))
of H. Mäkinen et al., "Optimization of contrast and dose in x-ray
phase-contrast tomography with a Talbot-Lau interferometer," *Biomed. Phys.
Eng. Express* 10, 045045 (2024),
[https://doi.org/10.1088/2057-1976/ad5206](https://doi.org/10.1088/2057-1976/ad5206),
in which this is the 35 kV water-container scan.

`real_data/extract_central_row.py` rebuilds the four files from the unpacked
images and needs only numpy. It takes the transaxial detector line through the
detector centre (column 768 of the 864 × 1536 raw images), keeps rows 10–852,
and shifts this window at each angle by the random sample offset recorded in
`scan.csv` (−10 to 10 pixels, applied during acquisition to reduce ring
artefacts), for both the object (`im`) and reference (`ref`) images:

```bash
python real_data/extract_central_row.py /path/to/raw   # folder with the .smv files and scan.csv
```

Synthetic data is built by `synthetic_data/create_data.py`.

## Reproducing the results

`--scenario` takes `full`, `sparse_angle`, `sparse_step` or `combined`
throughout. Every prior parameter can be overridden on the command line
(`--alpha_mu`, `--alpha_delta`, `--alpha_eps`, `--alpha_joint`, `--tv_beta`,
`--lambda0`), so an operating point can be changed without editing the source.
The sampling scripts default to 200 samples and 200 warmup iterations for quick
runs; the commands below use the settings of the paper (see
[Runtimes](#runtimes)).

### Synthetic study

Data are simulated at 512² with a bilinear projector and reconstructed at 256²
with a nearest-neighbour projector, so the reconstruction never inverts the
operator that generated the data.

**1. Simulate the data.** Builds the phantom and writes `synthetic_data.h5`.

```bash
python synthetic_data/create_data.py
```

**2. Select the regularization parameters.** The smoothing width β is fixed
first on full data; the channel weights are then swept on every acquisition
condition, the amplitude weight λ₀ on the two extremes, and the coupling weight
on every condition.

```bash
python synthetic_data/recon/sweep_tv_beta.py --stage alpha_beta   # smoothing beta, on full data

for s in full sparse_angle sparse_step combined; do               # channel weights, per scenario
    python synthetic_data/recon/sweep_tv_beta.py --stage alpha_beta --betas 3e-4 --scenario $s
done
python synthetic_data/recon/select_alpha.py                       # -> 100 / 316.2 / 56.23

for s in full combined; do                                        # amplitude weight lambda0
    python synthetic_data/recon/sweep_tv_beta.py --stage lambda0 --scenario $s --tv_beta 3e-4 \
        --alpha_mu 100 --alpha_delta 316.2 --alpha_eps 56.23
done

for s in full sparse_angle sparse_step combined; do               # joint weight, per scenario
    python synthetic_data/recon/tune_alpha_joint.py --scenario $s \
        --alpha_mu 100 --alpha_delta 316.2 --alpha_eps 56.23
done
```

The channel weights are chosen to minimize relative error *averaged over the
four acquisition conditions*, not on full data alone. The full-scan optima
(`177.8 / 1778 / 100`) over-regularize the undersampled cases: `alpha_delta =
1778` doubles the phase channel’s error under combined undersampling. This is
the tuning effect discussed in the paper, so `tune_alpha.py`, a convenience
script that tunes on full data only, deliberately reports those full-scan
values rather than the selected ones. Each sweep reports only its own
condition's optimum, which is why they disagree with one another and with the
paper; `select_alpha.py` combines the four grids and reports the weight with the
lowest mean error per channel, together with how far it sits from each
condition's own optimum (at most 1.5 %, 2.4 % and 3.9 % for μ, δ and ε). All
selected values are fixed in `synthetic_data/recon/run_scenarios.py`.

**3. MAP reconstruction.** Reconstructs FBP, TV and JTV for all four conditions
at the selected parameters, and draws the comparison figure.

```bash
python synthetic_data/recon/run_scenarios.py
python synthetic_data/recon/fig_map_comparison.py
```

**4. Posterior sampling.** One run per prior. Repeat with `--scenario` for the
other three conditions; each writes a `nuts_samples.pickle` archive.

```bash
python synthetic_data/uq/nuts_synthetic.py --phantom multicontrast --scenario full \
    --alpha_joint 0 --num_samples 1000 --warmup_steps 500        # TV
python synthetic_data/uq/nuts_synthetic.py --phantom multicontrast --scenario full \
    --alpha_joint 29.55 --num_samples 1000 --warmup_steps 500    # joint TV
```

**5. Posterior summaries, calibration and figures.** Sampling writes only the
archive. Post-processing reduces it to the `posterior_mean_std.npz` summary that
the calibration table and the UQ figure read, one scenario and one prior per
call, so repeat it for every scenario and prior to be shown. Note that
`post_process_nuts.py` defaults to `--phantom inclusion`, while the rest of the
synthetic pipeline uses `multicontrast`.

```bash
python synthetic_data/uq/post_process_nuts.py --phantom multicontrast --scenario full --prior tv
python synthetic_data/uq/post_process_nuts.py --phantom multicontrast --scenario full --prior jtv
python synthetic_data/uq/uq_calibration.py
python synthetic_data/uq/fig_uq_comparison.py
```

### Experimental study

**1. Select the regularization parameters.** β and λ₀ are chosen first, by a
grid search on sparse-angle data (the sweep's default `--scenario`) scored by
SSIM against the 361-view FBP; then the channel weights, by held-out angles;
then the coupling weight.

```bash
python real_data/recon/sweep_tv_beta_lambda0.py --betas 0.01 0.03 0.1 \
    --lambda0s 1e-6 1e-4 1e-3 1e-2 1e-1          # smoothing beta, amplitude weight lambda0
python real_data/recon/cv_alpha_real.py          # channel weights, by held-out angles
python real_data/recon/cv_alpha_real_joint.py \
    --alpha_mu 5.62 --alpha_delta 10 --alpha_eps 1.78    # held-out error vs. joint weight
python real_data/recon/tune_alpha_joint_ssim_sparse.py \
    --alpha_mu 5.62 --alpha_delta 10 --alpha_eps 1.78    # joint weight, by SSIM on sparse angles
```

`cv_alpha_real.py` reports its held-out minimum at `alpha_delta = 17.78`; the
paper uses `alpha_delta = 10`, the lower end of the range within 0.42 % of that
minimum, because it samples better. The held-out error barely changes with
`alpha_joint`, so its value (1) comes from the SSIM sweep. As in the synthetic
study, reading a choice off these sweeps is a manual step: β and λ₀ are
hard-coded in `cv_alpha_real.py` (`TV_BETA`, `LAMBDA0`) as well as in
`real_data/recon/run_scenarios.py`, so changing either means editing both.

**2. MAP reconstruction.** FBP, TV and JTV across the four acquisition
conditions, and the comparison figure.

```bash
python real_data/recon/run_scenarios.py
python real_data/recon/fig_map_comparison.py
```

**3. Posterior sampling.** One run per prior; repeat with `--scenario` for the
other three conditions.

```bash
python real_data/uq/nuts_real.py --scenario full --alpha_joint 0 --num_samples 1000    # TV
python real_data/uq/nuts_real.py --scenario full --num_samples 1000                    # joint TV
```

**4. Posterior summaries and figures.** `uq_summary.py` is the exception to the
post-processing step: it reads the sample archives directly and needs no
`.npz`. The UQ figure does need them.

```bash
python real_data/uq/post_process_real.py --scenario full --prior tv
python real_data/uq/post_process_real.py --scenario full --prior jtv
python real_data/uq/uq_summary.py
python real_data/uq/fig_uq_comparison.py
```

### Forward-model validation

Round-trips the measured data through retrieval, reconstruction and the assumed
channel models, and compares against the measurement.

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
per run and are not tracked; post-processing reduces each to a small
`posterior_mean_std.npz` summary stored alongside it.

## License

MIT (see [LICENSE](LICENSE)).
