# A Finite-Volume-Compatible Neural WENO Reconstruction Framework with Shared Learned Smoothness Closure

This repository is the reviewer-facing reproducibility snapshot for **WENO-SR**, a shared neural smoothness-ratio closure for fifth- and seventh-order finite-volume WENO reconstruction. The networks are trained end to end through complete finite-volume advection trajectories and are then transferred, without Euler training data, to one-, two-, and three-dimensional compressible-flow calculations.

The archive contains the exact selected checkpoints, immutable snapshots of the training and inference sources, publication figures and compact error tables, and one documented reproduction entry point per manuscript experiment. Large numerical fields are intentionally excluded so that the repository remains suitable for GitHub.

## Method at a glance

- **Training equation:** periodic scalar advection with velocities `a in {-1,+1}` and exact finite-volume cell averages at every completed time step.
- **Training profiles:** smooth Fourier waves, Gaussian pulses, square waves, triangular pulses, semi-ellipses, multiple jumps, and random composites; profile parameters are sampled on the fly.
- **Propagation distances:** 2, 8, 64, 128, 256, 512, and 1024 cells, sampled with equal cycle frequency.
- **WENO5 network:** `5 -> 10 -> 6 -> 6 -> 3`, coupled to SSPRK3.
- **WENO7 network:** `6 -> 24 -> 16 -> 16 -> 4`, coupled in this study to a fourth-order SSP Runge--Kutta method.
- **Equivariance:** every learned result uses the same reflection average used during training, `0.5 [M(x) + P M(Px)]` (with the four-output WENO7 permutation where appropriate).
- **Transfer:** the networks are trained only on one-dimensional scalar advection; the Euler shock tubes, vortex, two-dimensional Riemann problems, double Mach reflection, and shock--bubble interactions are out of distribution.

The finite-volume state, characteristic projection, Riemann solver, fluxes, and time update remain double precision in every reported method. `WENO5-SR-FP32` changes only the MLP parameters and hidden activations to FP32.

## Selected models

| Model | Training step | Parameter dtype | SHA-256 |
|---|---:|---|---|
| WENO5-SR | 12,250 | FP64 | `368759415a7dcf5567af76db06fd49c0b89260150f0fab7d9ada7f067ab3f74e` |
| WENO5-SR-FP32 | 16,500 | FP32 | `c88441a950b91713353685edc0aa4debcb848fdddb1ba1b9442dd893a40600bc` |
| WENO7-SR | 16,750 | FP64 | `0a55fd07a87e73b28e1c471991322dc256ddebccdbfdf2d5ba3722ae8dde3d93` |

The files and full architecture/precision notes are in [`models/`](models/).

## Repository layout

```text
paper/          current traceable manuscript draft
models/         the three selected checkpoints and model card
code/snapshot/  verbatim training, inference, baseline, and plotting sources
experiments/    one documented directory per published experiment
scripts/        release construction and integrity verification
```

The code snapshot preserves the original repository-relative module layout. Run commands from `code/snapshot/` or use the `run.sh` and `plot.sh` wrappers in each experiment directory.

## Experiments

| ID | Benchmark | Principal configuration |
|---:|---|---|
| 01 | [Long-time GSTE advection](experiments/01_gste_long_time_advection/) | N=200, t=10, CFL 0.6 |
| 02 | [Sod shock tube](experiments/02_sod_shock_tube/) | N=100, t=0.2, CFL 0.8 |
| 03 | [Lax shock tube](experiments/03_lax_shock_tube/) | N=200, t=1.3, CFL 0.8 |
| 04 | [Titarev--Toro](experiments/04_titarev_toro/) | N=1001/2000, t=5, CFL 0.8 |
| 05 | [Isentropic vortex](experiments/05_isentropic_vortex/) | 25^2 through 200^2, t=2, CFL 0.4 |
| 06 | [Riemann C.3](experiments/06_riemann_c3/) | 400^2, t=0.50, CFL 0.4 |
| 07 | [Riemann C.4](experiments/07_riemann_c4/) | 400^2, t=0.25, CFL 0.4 |
| 08 | [Riemann C.5](experiments/08_riemann_c5/) | 400^2, t=0.23, CFL 0.4 |
| 09 | [Riemann C.6](experiments/09_riemann_c6/) | 400^2, t=0.30, CFL 0.4 |
| 10 | [Double Mach reflection](experiments/10_double_mach_reflection/) | 1200x300, t=0.2, CFL 0.4 |
| 11 | [Planar shock--bubble, Ma=1.22](experiments/11_shock_bubble_2d_ma122/) | 1000x396, t=6e-4 |
| 12 | [Planar shock--bubble, Ma=3](experiments/12_shock_bubble_2d_ma30/) | 1000x396, t=1e-4 |
| 13 | [Three-dimensional shock--bubble, Ma=3](experiments/13_shock_bubble_3d_ma30/) | 224x88x88, t=1e-4 |
| 14 | [Mixed-precision timing](experiments/14_mixed_precision_timing/) | C.4 on 400^2, three repeats |

Each directory records the equations, complete initial condition, boundary treatment, quadrature, grid, CFL, final time, flux, selected models, result interpretation, plotting convention, exact commands, and artifact provenance.

## Installation and verification

Python 3.11 or newer and a CUDA-capable PyTorch/Warp installation are recommended. A minimal environment can be installed with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Verify the archive before running a calculation:

```bash
python3 scripts/verify_release.py
```

The verifier checks checkpoint hashes, tensor shapes and dtypes, reflection-symmetric forward maps, immutable source/artifact provenance, PDF integrity, experiment completeness, the 50 MB file limit, and the absence of raw numerical fields.

To reproduce one benchmark, for example GSTE:

```bash
cd experiments/01_gste_long_time_advection
DEVICE=cuda bash run.sh
bash plot.sh
```

Formal multidimensional calculations require substantial GPU time and memory. Each wrapper redirects regenerated fields and plots to the ignored `runs/` directory of that experiment, leaving `code/snapshot/` immutable.

## Data and provenance policy

- Publication figures and compact CSV/TEX summaries are included.
- Raw one-, two-, and three-dimensional numerical states are not included.
- Only the three paper-selected checkpoints are distributed.
- Failed training branches, exploratory probes, logs, process files, caches, and intermediate checkpoints are excluded.
- Third-party figures are not redistributed; the manuscript citation remains the authoritative source.
- [`MANIFEST.sha256`](MANIFEST.sha256) records every release file after verification.

The current traceable draft is available as [`paper/WENO_SR_current_draft.pdf`](paper/WENO_SR_current_draft.pdf). It is included for review context and is not presented as the final accepted manuscript.

## Authors

Ruijie Liu, Yuxuan Xia, Jianchuan Yang, Riccardo Dematte, and Jidong Zhao.
