# Order-matched WENO-Z verification

This directory is an isolated comparison package. It does not modify the
trusted JS/SR solvers, their raw results, or the paper.

## Formulation

The nonlinear weights are

```text
alpha_k = d_k * (1 + (tau / (beta_k + epsilon))^p_Z)
omega_k = alpha_k / sum_j(alpha_j)
```

The order-matched powers are `p_Z=2` for WENO5 and `p_Z=3` for WENO7.

with `tau5 = abs(beta0 - beta2)` for WENO5.  For WENO7, the numerical
experiments after Remark 10 of Castro--Costa--Don use the Table 2 optimal
indicator
`tau7_opt = abs(-beta0 - 3*beta1 + 3*beta2 + beta3)`, not the generic
Theorem 2 indicator `abs(beta0 - beta1 - beta2 + beta3)`.  The WENO7 kernels
use the paper's direction-wise `epsilon=h^3`; their beta values carry a common
factor 240, so the kernel denominator uses the equivalent `240*h^3`.

WENO5 uses the paper's direction-wise `epsilon=h^2`; WENO7 uses
`epsilon=h^3`. This distinction from the fixed-epsilon weight-analysis
experiment is recorded in `FORMULA_AUDIT.md`.
All Euler tests use characteristic reconstruction and no ENO cutoff. The
one- and two-dimensional tests use HLLC. The three-dimensional shock--bubble
test follows the trusted three-dimensional Warp path and uses EVILIN.
WENO5 is paired with SSPRK3 and WENO7 with the four-stage downwind Shu--RK4
method.

For WENO7, the Jiang--Shu indicators are evaluated in an algebraically
equivalent nonnegative sum-of-squares form. This avoids catastrophic
cancellation in nearly constant characteristic fields with large energy
components; it does not clip the indicators or alter the conserved state.

## Completed tests

| Test | Configuration | WENO5-Z L1 | WENO7-Z L1 |
|---|---|---:|---:|
| GSTE | N=200, t=10, CFL=0.6 | 4.198206e-2 | 2.928891e-2 |
| Sod | N=100, t=0.2, CFL=0.8 | 4.495034e-3 | 5.067383e-3 |
| Lax | N=200, t=1.3, CFL=0.8 | 5.399423e-3 | 6.104028e-3 |
| Titarev--Toro | N=1001, t=5, CFL=0.8 | 3.666953e-2 | 2.192848e-2 |
| Titarev--Toro | N=2000, t=5, CFL=0.8 | 4.125768e-3 | 2.379461e-3 |

GSTE uses the exact periodic finite-volume solution. Sod and Lax errors use
15-point Gauss exact cell averages. Titarev--Toro errors are measured against
the existing WENO5-JS `N=10001` reference interpolated to the tested grid.

## Multidimensional tests

| Test | Configuration | WENO5-Z | WENO7-Z |
|---|---|---|---|
| Riemann C3 | 400x400, t=0.5, CFL=0.4 | not requested | complete, 1201 steps |
| Riemann C4 | 400x400, t=0.25, CFL=0.4 | not requested | complete, 639 steps |
| Riemann C5 | 400x400, t=0.23, CFL=0.4 | not requested | complete, 636 steps |
| Riemann C6 | 400x400, t=0.30, CFL=0.4 | not requested | complete, 589 steps |
| Double-Mach reflection | 1200x300, t=0.2, CFL=0.4 | complete, 3215 steps | complete, 3177 steps |
| Shock--bubble, Ma=1.22 | 1000x396, t=6e-4, CFL=0.4 | complete, 7336 steps | complete, 7329 steps |
| Shock--bubble, Ma=3.0 | 1000x396, t=1e-4, CFL=0.4 | complete, 2467 steps | complete, 2483 steps |
| 3-D shock--bubble, Ma=3.0 | 224x88x88, t=1e-4, CFL=0.25 | complete, 840 steps | complete, 832 steps |

The superseded outputs produced before the formula audit were deleted before
this complete rerun. No CFL reduction, limiter, positivity repair, solver
substitution, or ENO cutoff is used in the corrected runs.

## Figures

- `figures/gste/N200_t10_cfl06/`
- `figures/riemann_1d/sod/`
- `figures/riemann_1d/lax/`
- `figures/titarev_toro_cfl08/N1001x10/`
- `figures/titarev_toro_cfl08/N2000x10/`
- `figures/riemann/c3/` through `figures/riemann/c6/`
- `figures/double_mach/`
- `figures/shockbubble_2d/ma122/`
- `figures/shockbubble_2d/ma30/`
- `figures/shockbubble_3d/N224x88x88/`

Every plot includes the existing JS and SR results on the same axes. PNG and
PDF versions are provided. Error CSV files are under `tables/`.

## Reproduction

```bash
python3 -m weno_z_borges_p2_results.run_gste --device cuda
python3 -m weno_z_borges_p2_results.plot_gste

python3 -m weno_z_borges_p2_results.run_riemann_1d \
  --problem sod --nx 100 --ny 10 --cfl 0.8 \
  --out-tag N100_t020_cfl08 --device cuda
python3 -m weno_z_borges_p2_results.plot_riemann_1d_zoom \
  --problem sod --nx 100 --ny 10 --cfl 0.8 --input-tag N100_t020_cfl08 \
  --output-tag N100_t020_cfl08 --paper-style

python3 -m weno_z_borges_p2_results.run_riemann_1d --problem lax --device cuda
python3 -m weno_z_borges_p2_results.plot_lax_paper

python3 -m weno_z_borges_p2_results.run_titarev_toro --nx 1001 --device cuda
python3 -m weno_z_borges_p2_results.plot_titarev_toro --nx 1001

python3 -m weno_z_borges_p2_results.run_titarev_toro --nx 2000 --device cuda
python3 -m weno_z_borges_p2_results.plot_titarev_toro --nx 2000

for case in c3 c4 c5 c6; do
  python3 -m weno_z_borges_p2_results.run_quadrant \
    --case "$case" --nx 400 --ny 400 --cfl 0.4 --device cuda:0
done
python3 -m weno_z_borges_p2_results.plot_riemann

python3 -m weno_z_borges_p2_results.run_double_mach \
  --methods weno5_z_p2,weno7_z_p3 --nx 1200 --ny 300 \
  --cfl 0.4 --t-end 0.2 --device cuda:0
python3 -m weno_z_borges_p2_results.plot_double_mach

for method in weno5_z_p2 weno7_z_p3; do
  python3 -m weno_z_borges_p2_results.run_shockbubble_2d \
    --case ma122 --method "$method" --nx 1000 --ny 396 --cfl 0.4 \
    --t-end 6e-4 --device cuda:0
  python3 -m weno_z_borges_p2_results.run_shockbubble_2d \
    --case ma30 --method "$method" --nx 1000 --ny 396 --cfl 0.4 \
    --t-end 1e-4 --device cuda:0
done
python3 -m weno_z_borges_p2_results.plot_shockbubble_2d --case ma122
python3 -m weno_z_borges_p2_results.plot_shockbubble_2d --case ma30

python3 -m weno_z_borges_p2_results.run_shockbubble_3d \
  --method weno5_z_p2 --nx 224 --ny 88 --nz 88 \
  --cfl 0.25 --t-end 1e-4 --device cuda:0
python3 -m weno_z_borges_p2_results.run_shockbubble_3d \
  --method weno7_z_p3 --nx 224 --ny 88 --nz 88 \
  --cfl 0.25 --t-end 1e-4 --device cuda:0
python3 -m weno_z_borges_p2_results.plot_shockbubble_3d
python3 -m weno_z_borges_p2_results.plot_shockbubble_3d_linecuts \
  --x 0.085 0.097 0.114
```

## Discarded p_Z=1 shock-bubble branch

The formal WENO7-Z p_Z=1 two-dimensional shock-bubble runs became non-finite
for both Ma=3.0 (step 455) and Ma=1.22 (step 3462). Their generated figures
were deleted. Raw failure manifests and logs remain in
`weno_z_borges_p1_results/` for reproducibility. The order-matched
shock--bubble results above were generated independently with `p_Z=3`.
