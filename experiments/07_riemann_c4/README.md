# Two-dimensional Riemann configuration C.4

## Purpose

Two-dimensional compressible Euler equations with gamma=1.4. This benchmark is part of the out-of-distribution validation suite for the shared learned smoothness-ratio closure.

## Configuration

The domain [0,1]^2 uses a 400x400 finite-volume mesh, outflow boundaries, characteristic HLLC fluxes, CFL 0.4, and t=0.25. Initial discontinuities are aligned with x=0.5 and y=0.5.

## Initial condition

States V1--V4 occupy the upper-right, upper-left, lower-left, and lower-right quadrants: V1=(1.1,0,0,1.1); V2=(0.5065,0.8939,0,0.35); V3=(1.1,0.8939,0.8939,1.1); V4=(0.5065,0,0.8939,0.35). Tensor-product 15x15 Gauss--Legendre quadrature constructs the initial finite-volume averages.

## Methods

WENO5-JS--RK3, WENO5-SR--RK3, WENO5-SR-FP32--RK3, WENO7-JS--RK4, WENO7-Z--RK4, and WENO7-SR--RK4. The learned methods use the selected WENO5-SR FP64 checkpoint at step 12,250, the independently trained WENO5-SR FP32 checkpoint at step 16,500, and the WENO7-SR FP64 checkpoint at step 16,750. Reflection-symmetrised inference is used in every learned reconstruction.

## Results

All methods reproduce the symmetric lens-shaped interaction; the learned closures retain sharper internal contacts without changing the principal wave topology.

The compact numerical values used by the manuscript are retained in [`tables/`](tables/), while the publication-ready result is retained in [`figures/`](figures/).

## Plot construction

Pressure is shown in colour and density as black contours. Every method in this configuration shares one pressure normalisation and 19 equally spaced density levels on [0.64,1.92]. Plot coordinates are translated by (0.5,0.5).

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
