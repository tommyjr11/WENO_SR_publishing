# Three-dimensional Mach-3 shock--bubble interaction

## Purpose

Three-dimensional single-material compressible Euler equations with gamma=1.4. This benchmark is part of the out-of-distribution validation suite for the shared learned smoothness-ratio closure.

## Configuration

The domain [0,0.225]x[0,0.089]x[0,0.089] uses 224x88x88 cells, transmissive boundaries, characteristic reconstruction, EVILIN fluxes, CFL 0.25, and t=1.0e-4. Classical JS context is also available on 448x176x176. The reference is WENO7--ADER4 on 1120x440x440 with the same EVILIN flux.

## Initial condition

The shock starts at x=0.005. The post-shock state is (rho,u,v,w,p)=(4.975714,736.911,0,0,1.047025e6), air is (1.29,0,0,0,101325), and the spherical light region is (0.214,0,0,0,101325), centred at (0.035,0.0445,0.0445) with radius 0.025. Initial conservative cell averages use 3x3x3 Gauss quadrature.

## Methods

WENO5-JS--RK3, WENO5-Z--RK3, WENO5-SR--RK3, WENO5-SR-FP32--RK3, WENO7-JS--RK4, WENO7-Z--RK4, and WENO7-SR--RK4. The learned methods use the selected WENO5-SR FP64 checkpoint at step 12,250, the independently trained WENO5-SR FP32 checkpoint at step 16,500, and the WENO7-SR FP64 checkpoint at step 16,750. Reflection-symmetrised inference is used in every learned reconstruction.

## Results

WENO7-SR has the smallest 224x88x88 integrated density error on all three central-plane cuts. At x=0.085 it reduces the WENO7-JS error by 21.9%; the FP64 and FP32 WENO5-SR fields are visually and quantitatively very close. The learned improvements are concentrated around the rolled-up, under-resolved interface.

The compact numerical values used by the manuscript are retained in [`tables/`](tables/), while the publication-ready result is retained in [`figures/`](figures/).

## Plot construction

Volume renderings and central-z mock-schlieren use common transfer functions and normalisations. Density profiles at x=0.085, 0.097, and 0.114 are compared to the high-resolution reference; 2:1 data are symmetrically averaged and the aligned 5:1 reference uses coincident cell centres.

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
