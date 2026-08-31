# Long-time advection of the GSTE profile

## Purpose

One-dimensional scalar advection, u_t + u_x = 0. This benchmark is part of the out-of-distribution validation suite for the shared learned smoothness-ratio closure.

## Configuration

The periodic domain is [-1, 1], discretised with N=200 finite-volume cells. The solution is advanced to t=10 at CFL 0.6, corresponding to five complete domain traversals and 1,667 time steps. Initial and exact cell averages use 15-point Gauss--Legendre quadrature.

## Initial condition

The standard GSTE composite contains a narrow Gaussian packet on (-0.8,-0.6), a unit square wave on (-0.4,-0.2), a triangular pulse on (0,0.2), and a semi-ellipse on (0.4,0.6). The Gaussian and semi-ellipse use the three-point 1:4:1 smoothing prescribed by the benchmark. The exact solution is the periodic translation u(x,t)=u_0(x-t).

## Methods

WENO5-JS--RK3, WENO5-Z--RK3, WENO5-SR--RK3, WENO5-SR-FP32--RK3, WENO7-JS--RK4, WENO7-Z--RK4, and WENO7-SR--RK4. The learned methods use the selected WENO5-SR FP64 checkpoint at step 12,250, the independently trained WENO5-SR FP32 checkpoint at step 16,500, and the WENO7-SR FP64 checkpoint at step 16,750. Reflection-symmetrised inference is used in every learned reconstruction.

## Results

All learned methods remain stable for the full trajectory although GSTE is not a training profile. WENO7-SR gives the lowest L1 error, 2.342e-2, improving on WENO7-JS by 21.3% and WENO7-Z by 36.3%. The WENO5-SR FP32 model is the most accurate fifth-order variant with L1=3.645e-2.

The compact numerical values used by the manuscript are retained in [`tables/`](tables/), while the publication-ready result is retained in [`figures/`](figures/).

## Plot construction

The exact translated profile and all seven finite-volume solutions are drawn on the same axes. Method colours and line styles are fixed globally; no method-specific axis limits or filtering is applied.

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
