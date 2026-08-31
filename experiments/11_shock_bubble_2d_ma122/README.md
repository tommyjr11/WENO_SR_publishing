# Planar Mach-1.22 shock--helium-bubble interaction

## Purpose

Two-dimensional single-gamma compressible Euler equations with gamma=1.4. This benchmark is part of the out-of-distribution validation suite for the shared learned smoothness-ratio closure.

## Configuration

The domain [0,0.225]x[0,0.089] uses 1000x396 cells, transmissive boundaries, characteristic HLLC fluxes, CFL 0.4, and t=6.0e-4. The comparison reference is an independent WENO7-JS--RK4 solution on 2000x791 cells.

## Initial condition

A planar Mach-1.22 shock starts at x=0.005. Air has rho=1.29 and p=101325; the circular helium region has rho=0.214, the same pressure, radius 0.025, and centre (0.035,0.0445). Rankine--Hugoniot relations determine the post-shock density and pressure, with u_post=110.6273. Cell averages use 15x15 quadrature.

## Methods

WENO5-JS--RK3, WENO5-Z--RK3, WENO5-SR--RK3, WENO5-SR-FP32--RK3, WENO7-JS--RK4, WENO7-Z--RK4, and WENO7-SR--RK4. The learned methods use the selected WENO5-SR FP64 checkpoint at step 12,250, the independently trained WENO5-SR FP32 checkpoint at step 16,500, and the WENO7-SR FP64 checkpoint at step 16,750. Reflection-symmetrised inference is used in every learned reconstruction.

## Results

The WENO7-SR vortex-core locations and rolled-up interface agree closely with the 2000x791 reference. It has the smallest local density L1 error on all three reported cuts, reducing WENO7-JS errors by 65.2%, 51.1%, and 35.5%.

The compact numerical values used by the manuscript are retained in [`tables/`](tables/), while the publication-ready result is retained in [`figures/`](figures/).

## Plot construction

The full-domain mock-schlieren panels use one shared normalisation. Density cuts are reported at y=0.01900, y=0.04525, and x=0.12566; the fine reference is conservatively restricted before L1 evaluation.

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
