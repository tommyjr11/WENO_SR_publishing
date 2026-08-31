# Lax shock tube

## Purpose

One-dimensional compressible Euler equations with gamma=1.4. This benchmark is part of the out-of-distribution validation suite for the shared learned smoothness-ratio closure.

## Configuration

The domain [-5,5] uses N=200 finite-volume cells, transmissive boundaries, characteristic reconstruction, HLLC fluxes, CFL 0.8, and final time t=1.3.

## Initial condition

The initial interface is at x=0. Primitive states are (rho,u,p)=(0.445,0.698,3.528) on the left and (0.5,0,0.571) on the right. The reference is the exact finite-volume Riemann solution.

## Methods

WENO5-JS--RK3, WENO5-Z--RK3, WENO5-SR--RK3, WENO5-SR-FP32--RK3, WENO7-JS--RK4, WENO7-Z--RK4, and WENO7-SR--RK4. The learned methods use the selected WENO5-SR FP64 checkpoint at step 12,250, the independently trained WENO5-SR FP32 checkpoint at step 16,500, and the WENO7-SR FP64 checkpoint at step 16,750. Reflection-symmetrised inference is used in every learned reconstruction.

## Results

WENO7-SR gives the lowest density L1 error, 4.415e-3. WENO5-SR FP64 and FP32 are effectively coincident at 4.855e-3 and 4.856e-3, both clearly below the WENO5-JS and WENO5-Z errors.

The compact numerical values used by the manuscript are retained in [`tables/`](tables/), while the publication-ready result is retained in [`figures/`](figures/).

## Plot construction

Density, velocity, and pressure are compared to the exact solution. Insets use common windows around the contact and post-shock plateau; numerical values are shown as finite-volume samples rather than interpolated curves.

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
