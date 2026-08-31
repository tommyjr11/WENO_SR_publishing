# Periodic isentropic-vortex convergence

## Purpose

Two-dimensional compressible Euler equations with gamma=1.4. This benchmark is part of the out-of-distribution validation suite for the shared learned smoothness-ratio closure.

## Configuration

The periodic domain is [-10,10]^2. Meshes N=25, 50, 100, and 200 are evolved to t=2 at CFL 0.4. Initial and exact conservative cell averages use tensor-product 15x15 Gauss--Legendre quadrature.

## Initial condition

The free stream is rho=p=1 and (u,v)=(1,1). An isentropic vortex of strength beta=5 is centred at the origin and translates to (t,t). Periodic shortest-distance coordinates with period 20 define both the initial condition and exact solution.

## Methods

WENO5-JS--RK3, WENO5-Z--RK3, WENO5-SR--RK3, WENO5-SR-FP32--RK3, WENO7-JS--RK4, WENO7-Z--RK4, and WENO7-SR--RK4. The learned methods use the selected WENO5-SR FP64 checkpoint at step 12,250, the independently trained WENO5-SR FP32 checkpoint at step 16,500, and the WENO7-SR FP64 checkpoint at step 16,750. Reflection-symmetrised inference is used in every learned reconstruction.

## Results

The learned schemes retain high-order convergence. At N=200, WENO5-SR FP64 has rho-L1=2.701e-7 and WENO7-SR has 6.826e-8, both below their JS counterparts. WENO7-Z is the most accurate fine-grid smooth baseline, as expected from its critical-point construction.

The compact numerical values used by the manuscript are retained in [`tables/`](tables/), while the publication-ready result is retained in [`figures/`](figures/).

## Plot construction

Log--log L1 and L2 density errors are plotted against grid spacing with common reference slopes. Orders are computed between successive factor-two refinements.

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
