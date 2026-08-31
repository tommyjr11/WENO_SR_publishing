# Sod shock tube

## Purpose

One-dimensional compressible Euler equations with gamma=1.4. This benchmark is part of the out-of-distribution validation suite for the shared learned smoothness-ratio closure.

## Configuration

The domain [0,1] uses N=100 finite-volume cells, transmissive boundaries, characteristic reconstruction, HLLC fluxes, CFL 0.8, and t=0.2. The auxiliary transverse extent contains ten identical cells but is not part of the stated one-dimensional resolution.

## Initial condition

The discontinuity is at x=0.5. Primitive states are (rho,u,p)=(1,0,1) on the left and (0.125,0,0.1) on the right. Initial and exact finite-volume averages are evaluated with 15-point Gauss--Legendre quadrature.

## Methods

WENO5-JS--RK3, WENO5-Z--RK3, WENO5-SR--RK3, WENO5-SR-FP32--RK3, WENO7-JS--RK4, WENO7-Z--RK4, and WENO7-SR--RK4. The learned methods use the selected WENO5-SR FP64 checkpoint at step 12,250, the independently trained WENO5-SR FP32 checkpoint at step 16,500, and the WENO7-SR FP64 checkpoint at step 16,750. Reflection-symmetrised inference is used in every learned reconstruction.

## Results

WENO5-SR has the smallest density L1 error, 3.806e-3, which is 23.4% below WENO5-JS and 16.8% below WENO5-Z. WENO7-SR reaches 3.997e-3 and improves on both seventh-order classical baselines by approximately 19%.

The compact numerical values used by the manuscript are retained in [`tables/`](tables/), while the publication-ready result is retained in [`figures/`](figures/).

## Plot construction

Symbols denote numerical cell averages and a dense black curve denotes the exact pointwise Riemann solution. Enlarged panels show the post-rarefaction plateau, contact discontinuity, and shock with shared limits across methods.

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
