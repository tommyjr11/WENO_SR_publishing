# Warp 3-D WENO5-RK3 classical and WENO5-SR port

This isolated package ports only the active `HEOC_TR == 4` branch of
`ADER_TR_Project/ADER_TR_3D.cu`. It does not modify the trusted C++ sources,
initialization code, or reference result.

The numerical path is fixed to:

- Warp `float64` solver state, with fast math and floating-point contraction
  disabled;
- three-dimensional Euler equations;
- characteristic WENO5-JS reconstruction, or the V20 WENO5-SR MLP on every
  normal and transverse Gauss reconstruction;
- SSPRK3 time integration;
- EVILIN with the original `(right_state, left_state)` argument order and
  `1e-15` guards;
- three transmissive ghost layers, applied in x, y, z order;
- two transverse Gauss points per direction and four face points accumulated
  in flag order `1, 2, 3, 4`;
- the current Ma=3 shock-bubble initial condition and 3 x 3 x 3 Gauss cell
  averages.

No HLLC, MPI, ADER, or fused performance path is included here. Both optional
MLP paths keep the state, feature preprocessing, WENO normalization,
characteristic transforms, EVILIN flux, and SSPRK3 updates in `float64`. The
FP64 deployment evaluates the MLP in `float64`; the mixed deployment uses
`float32` only for MLP parameters, affine layers, activations, logits, and
softmax, then casts ratios back to `float64` before reflection averaging.

## Formal run

```bash
cd /home/ruijie/warp_ADER_WENO
python3 -u -m warp_weno5_3d_rk3.run_shockbubble_ma3 \
  --device cuda:0 \
  --out-dir warp_weno5_3d_rk3/runs/ma3_evilin_t00002 \
  --reference ADER_TR_Project/data/step_0140.bin
```

The command writes:

- `step_0140.bin` in the C++ binary schema;
- `dt_trace.csv` with every `dt`, maximum speed, and accumulated time;
- `comparison.json` with file, array, component, absolute-error, normalized
  L1, and ULP checks;
- `run_manifest.json` with configuration and positivity diagnostics.

Binary schema:

```text
uint32 nz, uint32 ny, uint32 nx, float64 time,
float64 primitive[nz][ny][nx][5]
```

## WENO5-SR V20 run

The formal FP64 model is V20 step 12,250. Deployment evaluates
`0.5*(M(x)+P*M(P*x))`, uses all four normal/Gauss heads, and leaves ENO cutoff
disabled:

```bash
python3 -u -m warp_weno5_3d_rk3.run_shockbubble_ma3_mlp \
  --device cuda:0 \
  --out-dir warp_weno5_3d_rk3/runs/ma3_v20_step012250_t00002
```

The model path may be changed with `--model`. The loader rejects incompatible
shapes, architecture metadata, or checkpoints that do not declare
reflection-symmetric deployment.

The selected mixed-FP32 model is V20 step 16,500:

```bash
python3 -u -m warp_weno5_3d_rk3.run_shockbubble_ma3_mlp_f32 \
  --device cuda:0 \
  --out-dir warp_weno5_3d_rk3/runs/ma3_v20_mlp_f32_step016500_t00002
```

The checkpoint dtype and mixed-precision metadata select the matching Warp
kernel path automatically. A float32 checkpoint is never silently promoted to
the FP64 MLP path.

## Verification

Run the lightweight tests:

```bash
python3 -u -m warp_weno5_3d_rk3.self_test --device cuda:0
python3 -u -m warp_weno5_3d_rk3.mlp_self_test --device cuda:0
python3 -u -m warp_weno5_3d_rk3.mlp_f32_self_test --device cuda:0
```

They check binary I/O, all transmissive ghost faces/edges/corners, uniform
flow preservation, and the C++-oracle initialization hash. The MLP test also
compares all four scalar reconstruction heads against the trusted Torch
deployment and advances a 3-D uniform flow. Add `--full` to the classical test
to repeat the complete 140-step reference comparison.

Compare two completed files independently:

```bash
python3 -m warp_weno5_3d_rk3.compare_bin \
  warp_weno5_3d_rk3/runs/ma3_evilin_t00002/step_0140.bin \
  ADER_TR_Project/data/step_0140.bin
```

The audited local classical run completed 140 steps at `t=2e-5` and passed byte for
byte. Both files have SHA-256:

```text
b977c367c59417155719ce53a7e787c7c3fb8088bc713bd4194c2d7b8218e7c9
```

All 8,673,280 primitive values match exactly: unequal count 0, maximum ULP 0,
maximum absolute error 0, and normalized L1 0. The audited MLP run completed
142 steps at the same final time with no NaNs, positive density and pressure,
and checkpoint SHA-256
`368759415a7dcf5567af76db06fd49c0b89260150f0fab7d9ada7f067ab3f74e`.

The three `224 x 88 x 88`, CFL `0.25`, `t=1e-4` runs are stored under
`runs/ma3_t0001_cfl025_N224x88x88/`. See its `SUMMARY.md` and shared-colorbar
midplane figure for paths, diagnostics, timings, and pairwise field
differences.

## Source-fidelity note

The trusted C++ normal reconstruction calls `LR=2` and then `LR=1` on the
same mutable local stencil. The first wrapper restores the stencil through
`R*(L*U)` before the second call. That round trip is mathematically an
identity but is not a bitwise identity. `normal_x_kernel`, `normal_y_kernel`,
and `normal_z_kernel` intentionally reproduce this order. Computing the two
sides independently is numerically close (about `1e-15` relative) but does
not reproduce the reference file bitwise.

## Small smoke run

```bash
python3 -u -m warp_weno5_3d_rk3.run_shockbubble_ma3 \
  --device cuda:0 --nx 12 --ny 10 --nz 10 --stop-step 1 \
  --out-dir warp_weno5_3d_rk3/runs/smoke --reference /dev/null
```
