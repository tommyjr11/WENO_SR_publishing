# Planar Mach-3 shock--helium-bubble interaction

## Purpose

Two-dimensional single-gamma compressible Euler equations with gamma=1.4. This benchmark is part of the out-of-distribution validation suite for the shared learned smoothness-ratio closure.

## Configuration

The domain [0,0.225]x[0,0.089] uses 1000x396 cells, transmissive boundaries, characteristic HLLC fluxes, CFL 0.4, and t=1.0e-4. The comparison reference is an independent WENO7-JS--RK4 solution on 2000x791 cells.

## Initial condition

A planar Mach-3 shock starts at x=0.005. Air has rho=1.29 and p=101325; the helium bubble has rho=0.214, radius 0.025, and centre (0.035,0.0445). The post-shock speed is 736.911 and Rankine--Hugoniot relations set rho and p. Cell averages use tensor-product 15x15 quadrature.

## Methods

WENO5-JS--RK3, WENO5-Z--RK3, WENO5-SR--RK3, WENO5-SR-FP32--RK3, WENO7-JS--RK4, WENO7-Z--RK4, and WENO7-SR--RK4. The learned methods use the selected WENO5-SR FP64 checkpoint at step 12,250, the independently trained WENO5-SR FP32 checkpoint at step 16,500, and the WENO7-SR FP64 checkpoint at step 16,750. Reflection-symmetrised inference is used in every learned reconstruction.

## Results

WENO7-SR is the most accurate 1000x396 method on all three selected cuts and reduces the WENO7-JS local density errors by 56.9%, 25.0%, and 31.9%. Both WENO5-SR precisions also improve on WENO5-Z on these cuts.

The compact numerical values used by the manuscript are retained in [`tables/`](tables/), while the publication-ready result is retained in [`figures/`](figures/).

## Plot construction

The full-domain mock-schlieren panels share one normalisation. Density cuts are taken at x=0.06750, x=0.10991, and y=0.02056 through complex interaction regions.

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
