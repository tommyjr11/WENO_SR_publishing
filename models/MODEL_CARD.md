# WENO-SR selected model card

This directory contains the three checkpoints used in every learned result in the manuscript. They are copied byte for byte from the original training runs. No checkpoint conversion or parameter editing is performed.

## Common training design

All models are initialised at the optimal linear weights and trained from scratch through complete differentiable finite-volume trajectories for periodic scalar advection. The numerical state remains attached to the autograd graph. Exact cell averages are available at every completed time step, so supervision is applied to evolved finite-volume states rather than isolated point values.

The profile generator samples smooth Fourier waves, Gaussian pulses, square waves, triangles, semi-ellipses, multiple jumps, and random composites. Velocities `-1` and `+1` are both included. Propagation distances are 2, 8, 64, 128, 256, 512, and 1024 cells, with one distance drawn from each shuffled cycle and batch sizes 32, 32, 16, 8, 4, 2, and 2. The primary training CFL is 0.5.

The loss compares the learned result to the exact cell averages and includes global and local non-regression guards relative to an order-matched Jiang--Shu solver using the same time stepper. Paper Euler configurations and the GSTE composite are not training samples.

## WENO5-SR FP64

- File: `weno5_sr_fp64_step012250.npz`
- SHA-256: `368759415a7dcf5567af76db06fd49c0b89260150f0fab7d9ada7f067ab3f74e`
- Source step: 12,250
- Architecture: `5 -> 10 -> 6 -> 6 -> 3`
- Parameter and activation dtype: FP64
- Time integrator in training: SSPRK3
- Feature vector:

  ```text
  [delta_0/delta_max,
   delta_1/delta_max,
   delta_2/delta_max,
   gamma_s,
   clip((log10(delta_max/q_scale)+16)/16, 0, 1)]
  ```

- Reflection map: `r_sym(x) = 0.5 [M(x) + P3 M(P5 x)]`
- Stored tensors: `5x10`, `10x6`, `6x6`, and `6x3` weights with matching biases.

The three outputs are shared beta-like badness ratios. They are converted to nonlinear weights with the appropriate linear weights for each face or Gauss point; the network does not directly output a reconstructed state.

## WENO5-SR FP32

- File: `weno5_sr_fp32_step016500.npz`
- SHA-256: `c88441a950b91713353685edc0aa4debcb848fdddb1ba1b9442dd893a40600bc`
- Source step: 16,500
- Architecture and features: identical to WENO5-SR FP64
- MLP parameter and hidden-activation dtype: FP32
- Feature preprocessing, reflection average, WENO normalisation, finite-volume state, characteristic transforms, Riemann solver, fluxes, and SSPRK3 update: FP64
- Reflection map: `r_sym(x) = 0.5 [M(x) + P3 M(P5 x)]`

This is an independently trained mixed-precision model, not a post-training cast of the FP64 checkpoint.

## WENO7-SR FP64

- File: `weno7_sr_fp64_step016750.npz`
- SHA-256: `0a55fd07a87e73b28e1c471991322dc256ddebccdbfdf2d5ba3722ae8dde3d93`
- Source step: 16,750
- Architecture: `6 -> 24 -> 16 -> 16 -> 4`
- Parameter and activation dtype: FP64
- Time integrator in training: fourth-order downwind TVD/SSP Runge--Kutta using `L` and `L_tilde`
- Feature vector: four normalised WENO7 delta measures, `gamma_s`, and the same clipped logarithmic scale feature used for WENO5
- Reflection map: `r_sym(x) = 0.5 [M(x) + P4 M(P6 x)]`
- Stored tensors: `6x24`, `24x16`, `16x16`, and `16x4` weights with matching biases.

The four outputs are shared badness ratios for the four WENO7 substencils. Face orientation and Gauss-point-specific optimal weights remain analytical and are not learned.

## Deployment invariant

The raw MLP is not the published inference operator. Every learned reconstruction must evaluate both the original and reflected feature vectors and average the permuted outputs. Omitting this operation changes the numerical method and does not reproduce the manuscript.
