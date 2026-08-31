# Mixed-precision WENO5 inference timing

## Purpose

Two-dimensional Euler configuration C.4 used as a fixed timing workload. This benchmark is part of the out-of-distribution validation suite for the shared learned smoothness-ratio closure.

## Configuration

The 400x400 C.4 calculation is timed after Warp compilation, model loading, initialisation, diagnostics, plotting, and file output are excluded. Each method is warmed up and then measured three times.

## Initial condition

The numerical problem is identical to experiment 07. Only the MLP parameter and activation precision changes: finite-volume states, characteristic transforms, HLLC fluxes, and SSPRK3 updates remain FP64 in all variants.

## Methods

WENO5-JS, WENO5-SR with an FP64 MLP, and WENO5-SR with an FP32 MLP.. The learned methods use the selected WENO5-SR FP64 checkpoint at step 12,250, the independently trained WENO5-SR FP32 checkpoint at step 16,500, and the WENO7-SR FP64 checkpoint at step 16,750. Reflection-symmetrised inference is used in every learned reconstruction.

## Results

Mean wall times are 6.605+/-0.032 s, 39.333+/-0.280 s, and 14.253+/-0.072 s for JS, FP64 MLP, and FP32 MLP. FP32 therefore accelerates learned inference by 2.76x while preserving the FP64 finite-volume solver.

The compact numerical values used by the manuscript are retained in [`tables/`](tables/), while the publication-ready result is retained in [`figures/`](figures/).

## Plot construction

No figure is used; the three-repeat statistics are reported as a LaTeX table.

The plotting scripts operate on regenerated local outputs and do not alter the archived publication figure. Vector PDF is retained whenever available.

## Reproduction

From this experiment directory:

```bash
DEVICE=cuda bash run.sh
bash plot.sh
```

The full calculation may require a CUDA GPU and substantial wall time. The wrappers redirect every solver-defined output root to this experiment's ignored `runs/` directory; the immutable code snapshot is therefore not populated with regenerated fields.

## Provenance and data policy

[`provenance.json`](provenance.json) records the original path, SHA-256 digest, and size of every archived figure and table. Raw multidimensional fields are deliberately excluded from this GitHub snapshot. The solver and plotting sources needed to regenerate them are preserved verbatim under [`../../code/snapshot/`](../../code/snapshot/).
