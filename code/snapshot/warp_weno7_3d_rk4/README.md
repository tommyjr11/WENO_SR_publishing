# Warp 3-D WENO7-JS/SR--RK4

This directory is an isolated Warp port of the constant-term spatial path in
`ADER_TR_Project/ADER_TR4_3D.cu`. It does not modify or replace the existing C++,
WENO5, ADER, or training implementations.

## Numerical method

- three-dimensional Euler equations in `float64`, stored as `(z,y,x,5)`;
- four transmissive or periodic ghost cells, applied in x, y, then z order;
- seven-cell characteristic WENO7-JS normal reconstruction;
- constant reconstruction value only (`Dorder=0`);
- two WENO7 Gauss values in each transverse direction, producing four flux
  points per face;
- EVILIN with the original `(right_state,left_state)` convention;
- the verified four-stage, fourth-order downwind TVD Runge--Kutta method;
- no HEOC/TR derivatives, history arrays, HLLC, or ADER terms.

The optional WENO7-SR path keeps the Euler state, characteristic projection,
EVILIN flux, and RK4 arithmetic in `float64`.  Its `6 -> 24 -> 16 -> 16 -> 4`
MLP replaces the JS weights in the normal reconstruction and in both
transverse Gauss reconstructions.  Every call uses the trained reflection-
equivariant average
`0.5 * (M(x) + P4 * M(P6*x))`; ENO cutoff remains disabled unless explicitly
requested.

The downwind operator uses the same reconstruction and swaps the two EVILIN
states. One RK step evaluates `L(U0)`, `tildeL(U0)`, `L(U1)`, `tildeL(U1)`,
`L(U2)`, and `L(U3)`.

## Verification

Run the independent checks with:

```bash
python3 -u -m warp_weno7_3d_rk4.self_test --device cuda:0
```

They cover binary round-tripping, scalar reflection identities, all four ghost
layers including edges and corners, constant-state preservation, exact reuse of
the trusted Ma=3 initialization, and reduction to the trusted two-dimensional
WENO7-JS operator for a z-invariant state.

## Formal run

```bash
python3 -u -m warp_weno7_3d_rk4.run_shockbubble_ma3 \
  --device cuda:0 \
  --nx 224 --ny 88 --nz 88 \
  --cfl 0.25 --t-end 2e-5 \
  --out-dir warp_weno7_3d_rk4/runs/ma3_js_rk4_t00002
```

The first Warp compilation can take several minutes. The run writes a C++
compatible `step_NNNN.bin`, `dt_trace.csv`, `run_manifest.json`, and central-z
density and mock-schlieren figures in PNG and PDF formats.

## WENO7-SR run

```bash
python3 -u -m warp_weno7_3d_rk4.run_shockbubble_ma3 \
  --device cuda:0 \
  --nx 224 --ny 88 --nz 88 \
  --cfl 0.25 --t-end 2e-5 \
  --model teacherfree_lab_weno7_rk4_distance_balanced_fast/runs/\
apost_weno7_rk4_distance_balanced_fast_4090_200k/checkpoints/\
model_step_016750.npz \
  --out-dir warp_weno7_3d_rk4/runs/ma3_sr_f64_step016750_rk4_t00002
```

Verify all four reconstruction heads against the training implementation with
`python3 -u -m warp_weno7_3d_rk4.verify_mlp`.  Once equal-time JS and SR files
exist, generate a shared-scale comparison with:

```bash
python3 -u -m warp_weno7_3d_rk4.plot_compare \
  --js JS_STEP.bin --sr SR_STEP.bin --out COMPARISON_STEM
```

## Three-plane order verification

The periodic isentropic vortex can be embedded independently in the xy, yz,
and xz planes. The inactive direction uses 10 cells, both active directions use
25, 50, 100, and 200 cells, and all initial and exact conserved cell averages
use 15x15 Gauss-Legendre quadrature.

```bash
python3 -u -m warp_weno7_3d_rk4.run_vortex_convergence \
  --device cuda:0 \
  --planes xy yz xz \
  --grids 25 50 100 200 \
  --inactive-cells 10 \
  --cfl 0.4 --t-end 2 --quadrature 15 \
  --out-dir warp_weno7_3d_rk4/runs/isentropic_vortex_cfl04
```

The command writes one binary and time-step trace per case, `convergence.csv`,
`SUMMARY.md`, and a PNG/PDF convergence plot.
