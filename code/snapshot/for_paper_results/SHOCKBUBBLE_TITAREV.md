# Shock--bubble and Titarev--Toro suite

This isolated suite uses the selected paper checkpoints from `config.py` and
does not modify any trusted solver or training directory.

## Numerical configuration

- WENO5-JS, WENO5-SR FP64, and WENO5-SR FP32-MLP/FP64-state.
- Characteristic reconstruction, HLLC flux, SSPRK3, transmissive boundaries,
  and no ENO cutoff.
- Every learned WENO5 call uses
  `0.5 * (M(x) + P M(Px))` through the established deployment adapters.
- Shock--bubble main grid: `1000x396`; WENO5-JS reference: `2000x791`.
- Shock--bubble initialization: `15x15` Gauss--Legendre cell averages.
- Titarev--Toro: `1001x8`, `t=5`, CFL `0.4`, with exact finite-volume initial
  averages.

## Overnight run

```bash
cd /home/ruijie/warp_ADER_WENO
nohup env GPU_ID=0 bash for_paper_results/run_shockbubble_titarev_overnight.sh \
  > for_paper_results/logs/shockbubble_titarev_overnight.log 2>&1 &
echo $! > for_paper_results/logs/shockbubble_titarev_overnight.pid
```

Monitor without attaching to the process:

```bash
bash for_paper_results/monitor_shockbubble_titarev.sh
```

The suite is restartable: completed `.npz` phases are skipped. Figures are
refreshed after every completed main-grid method. The shock--bubble line-cut
figure and CSV are generated once the `2000x791` reference is available.

The retained Mach-3 vortex-crossing line cuts, their precise local error
definition, and the numerical values used for paper discussion are recorded in
[`figures/shockbubble_selected_linecuts/ma30_complete_hierarchy/RESULTS.md`](figures/shockbubble_selected_linecuts/ma30_complete_hierarchy/RESULTS.md).

## Extended five-method run

The extended queue advances the shock--bubble problem to `t=6e-4`, adds
WENO7-JS-RK4 and WENO7-SR-RK4 to both benchmarks, and produces an additional
vortex-region mock-schlieren panel. It writes to a separate
`shockbubble_t0006` tree and can be restarted without repeating completed
fields:

```bash
setsid -f bash -c '
  echo $$ > for_paper_results/logs/shockbubble_t0006_titarev_all.pid
  exec env GPU_ID=0 bash for_paper_results/run_shockbubble_t0006_titarev_all.sh \
    >> for_paper_results/logs/shockbubble_t0006_titarev_all.log 2>&1
'
```

Monitor with:

```bash
bash for_paper_results/monitor_shockbubble_t0006_titarev_all.sh
```
