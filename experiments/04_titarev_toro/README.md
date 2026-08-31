# Titarev--Toro shock--entropy-wave interaction

## Purpose

One-dimensional compressible Euler equations with gamma=1.4. This benchmark is part of the out-of-distribution validation suite for the shared learned smoothness-ratio closure.

## Configuration

The domain [-5,5] is evaluated at N=1001 and N=2000, with transmissive boundaries, characteristic HLLC fluxes, CFL 0.8, and t=5.

## Initial condition

For x<-4.5 the primitive state is (1.515695,0.523346,1.805). Elsewhere it is (1+0.1 sin(20 pi x),0,1). The reference is a WENO5-JS calculation on N=10001, conservatively restricted to each reported mesh.

## Methods

WENO5-JS--RK3, WENO5-Z--RK3, WENO5-SR--RK3, WENO5-SR-FP32--RK3, WENO7-JS--RK4, WENO7-Z--RK4, and WENO7-SR--RK4. The learned methods use the selected WENO5-SR FP64 checkpoint at step 12,250, the independently trained WENO5-SR FP32 checkpoint at step 16,500, and the WENO7-SR FP64 checkpoint at step 16,750. Reflection-symmetrised inference is used in every learned reconstruction.

## Results

At N=1001, WENO5-SR has the lowest density L1 error, 1.682e-2. At N=2000, WENO7-SR is best at 1.849e-3, followed closely by WENO5-SR FP32 at 1.907e-3. The learned closures preserve substantially more of the post-shock entropy wave.

The compact numerical values used by the manuscript are retained in [`tables/`](tables/), while the publication-ready result is retained in [`figures/`](figures/).

## Plot construction

The full density profile and a post-shock enlargement use identical method colours at both resolutions. The high-resolution reference is restricted before error evaluation.

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
