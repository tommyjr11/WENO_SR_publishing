# WENO-SR paper results

This directory is the only location used for the new paper experiments.  It
does not modify training code, selected checkpoints, the ADER project, or old
validation outputs.

## Fixed methods

- `weno5_js`: classical WENO5-JS with SSPRK3.
- `weno5_sr_f64`: distance-balanced WENO5-SR step 12250, double-precision
  reflection-symmetric MLP evaluation.
- `weno5_sr_f32`: independently trained distance-balanced WENO5-SR step
  16500, FP32 MLP with an FP64 flow solver.
- `weno7_js`: classical WENO7-JS with fourth-order SSP Runge--Kutta.
- `weno7_sr_f64`: distance-balanced WENO7-SR step 16750.

Every Euler experiment uses characteristic reconstruction, HLLC, and no ENO
cutoff.  All learned methods use the same two-pass reflection-equivariant
evaluation as training, namely
`0.5 * (M(x) + P M(Px))`.  The outer signal speeds use the direct estimates
$S_L=\min(u_L-a_L,u_R-a_R)$ and
$S_R=\max(u_L+a_L,u_R+a_R)$.  Density, pressure, and the three HLLC
denominators use the common safeguard $\epsilon_{\rm HLLC}=10^{-16}$.

## Output policy

Raw solver output is stored under `raw/`, publication figures under
`figures/`, generated LaTeX tables under `tables/`, and command output under
`logs/`.  `manifest.json` records checkpoint hashes and the software/runtime
environment.

Run commands are added only after their corresponding smoke tests pass.  The
paper intentionally renders a visible TODO box whenever an expected validated
figure or table does not exist.

## Reproduction commands

Run from the repository root.  Every command writes only below this directory.

```bash
python3 -m for_paper_results.verify_hllc
python3 -m for_paper_results.smoke_euler
python3 -m for_paper_results.run_gste --t-end 10 --init-quadrature 15 \
  --weno5-cfl 0.6 --weno7-cfl 0.6 \
  --methods weno5_js,weno5_sr_f64,weno5_sr_f32,weno7_js,weno7_sr_f64
bash for_paper_results/run_sod_point_study.sh
python3 -m for_paper_results.run_vortex
bash for_paper_results/run_riemann_suite.sh
python3 -m for_paper_results.run_weno5_timing
python3 -m for_paper_results.run_double_mach
python3 -m for_paper_results.make_figures
python3 -m for_paper_results.make_riemann_figures
python3 -m for_paper_results.make_riemann_reference_figures
python3 -m for_paper_results.make_riemann_linecuts
python3 -m for_paper_results.make_double_mach_figures
```

`run_riemann_suite.sh` computes C.4, C.5, and C.6 sequentially and writes each
figure immediately after that configuration finishes.
The retained C.3 result is the previously validated q400 calculation at
`t=0.5`; the common figure generator reads it from `raw/q400/N400`.
The C.4--C.6 reference generator reads the completed classical
WENO5-JS--RK3 (1200^2) calculations from `raw/<case>/N1200`.  The line-cut
generator uses case-specific horizontal and vertical cuts through the resolved
wave systems and compares all five (400^2) solutions with those references.
The selected positions are recorded in `make_riemann_linecuts.py`, and the
corresponding profile errors are written to
`tables/riemann_linecut_errors_vs_weno5js_N1200.csv`.

The formal WENO7 double-Mach run uses the FP64 fused-Warp badness evaluator
(`--weno7-beta-backend warp`).  It is forward-equivalent to the standard
batched-PyTorch evaluator; the initial semi-discrete right-hand sides agree to
a relative discrepancy of \(9.3\times10^{-16}\).

For GSTE, the TT runner's fixed $\delta_{IS}<2.4$ detector is disabled because
it is a smooth-wave CFL diagnostic and the composite profile is discontinuous
at the initial time.  Completion time and finite output remain mandatory; raw
profiles and logs are retained for diagnosis.
