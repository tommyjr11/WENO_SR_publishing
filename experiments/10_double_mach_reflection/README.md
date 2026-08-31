# Double Mach reflection

## Purpose

Two-dimensional compressible Euler equations with gamma=1.4. This benchmark is part of the out-of-distribution validation suite for the shared learned smoothness-ratio closure.

## Configuration

The standard double-Mach reflection problem is computed on a 1200x300 mesh to t=0.2 with CFL 0.4, characteristic reconstruction, HLLC fluxes, 15-point finite-volume initialisation, and no ENO cutoff or state repair.

## Initial condition

A Mach-10 oblique shock initially intersects the lower wall at x=1/6 and forms a 60-degree angle with that wall. The moving post-shock state is imposed on the upper boundary consistently with the exact shock motion; the lower wall uses the standard inflow/reflection split.

## Methods

WENO5-JS--RK3, WENO5-Z--RK3, WENO5-SR--RK3, WENO5-SR-FP32--RK3, WENO7-JS--RK4, WENO7-Z--RK4, and WENO7-SR--RK4. The learned methods use the selected WENO5-SR FP64 checkpoint at step 12,250, the independently trained WENO5-SR FP32 checkpoint at step 16,500, and the WENO7-SR FP64 checkpoint at step 16,750. Reflection-symmetrised inference is used in every learned reconstruction.

## Results

All seven methods reach t=0.2 without non-finite values or negative density or pressure. WENO7-SR resolves the longest hierarchy of coherent secondary roll-ups along the slip line and near the wall while preserving the baseline shock geometry.

The compact numerical values used by the manuscript are retained in [`tables/`](tables/), while the publication-ready result is retained in [`figures/`](figures/).

## Plot construction

Separate WENO5 and WENO7 figures pair the full density field with the same 2.05<=x<=2.85, 0<=y<=0.55 enlargement. All methods share one density colour scale and 29 uniformly spaced black density contours. The third-party comparison image used for discussion in the draft is intentionally not redistributed here.

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
