from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import warp as wp

from . import kernels as K
from . import mlp_normal_scalar as MNS
from . import mlp_normal_scalar_f32 as MNS32
from . import mlp_transverse_scalar as MTS
from . import mlp_transverse_scalar_f32 as MTS32
from .binary_io import compare_steps, write_comparison, write_step
from .config import ShockBubbleConfig
from .mlp import MlpParameters, load_mlp_parameters
from .mlp_f32 import MlpFloat32Parameters, checkpoint_parameter_dtype, load_mlp_float32_parameters


class Weno5Rk3Solver:
    """Literal Warp port of the active C++ WENO5--RK3 CUDA path."""

    def __init__(
        self,
        config: ShockBubbleConfig,
        device: str = "cuda:0",
        strict_sync: bool = True,
        model: str | Path | None = None,
        eno_cutoff: bool = False,
    ):
        config.validate()
        self.config = config
        self.device = wp.get_device(device)
        self.strict_sync = strict_sync
        self.eno_cutoff = int(eno_cutoff)
        self.mlp: MlpParameters | MlpFloat32Parameters | None = None
        self.mlp_precision: str | None = None
        if model is not None:
            parameter_dtype = checkpoint_parameter_dtype(model)
            if parameter_dtype == np.dtype(np.float32):
                self.mlp = load_mlp_float32_parameters(model, self.device)
                self.mlp_precision = "float32"
            else:
                self.mlp = load_mlp_parameters(model, self.device)
                self.mlp_precision = "float64"

        shape = config.padded_shape
        self.q = wp.empty(shape, dtype=wp.float64, device=self.device)
        self.primitive = wp.empty(shape, dtype=wp.float64, device=self.device)
        self.q0 = wp.empty(shape, dtype=wp.float64, device=self.device)

        self.normal_left = wp.empty(shape, dtype=wp.float64, device=self.device)
        self.normal_right = wp.empty(shape, dtype=wp.float64, device=self.device)
        self.stage_left = wp.empty(shape, dtype=wp.float64, device=self.device)
        self.stage_right = wp.empty(shape, dtype=wp.float64, device=self.device)
        self.point_left = wp.empty(shape, dtype=wp.float64, device=self.device)
        self.point_right = wp.empty(shape, dtype=wp.float64, device=self.device)

        self.flux_x = wp.empty((config.nz, config.ny, config.nx + 1, 5), dtype=wp.float64, device=self.device)
        self.flux_y = wp.empty((config.nz, config.ny + 1, config.nx, 5), dtype=wp.float64, device=self.device)
        self.flux_z = wp.empty((config.nz + 1, config.ny, config.nx, 5), dtype=wp.float64, device=self.device)
        self.maximum = wp.zeros(1, dtype=wp.float64, device=self.device)

        self.time = float(config.t_start)
        self.step = 0
        self.dt_trace: list[dict[str, float | int]] = []

    def _mlp_kernel_inputs(self) -> list[object]:
        if self.mlp is None:
            raise RuntimeError("MLP kernel inputs requested for a classical run")
        return self.mlp.kernel_inputs() + [self.eno_cutoff]

    def _launch(self, kernel: object, dim: tuple[int, ...], inputs: list[object]) -> None:
        wp.launch(kernel, dim=dim, inputs=inputs, device=self.device)
        if self.strict_sync:
            wp.synchronize_device(self.device)

    def _normal_mlp(self, dim: tuple[int, int, int], direction: int) -> None:
        scalar_dim = dim + (5,)
        params = self._mlp_kernel_inputs()
        reconstruction_kernel = MNS.reconstruct_characteristic_mlp_kernel
        if self.mlp_precision == "float32":
            reconstruction_kernel = MNS32.reconstruct_characteristic_mlp_f32_kernel
        self._launch(
            reconstruction_kernel,
            scalar_dim,
            [self.q, self.stage_left, 2, direction] + params,
        )
        self._launch(
            reconstruction_kernel,
            scalar_dim,
            [self.q, self.stage_right, 1, direction] + params,
        )
        self._launch(
            MNS.characteristic_to_conserved_kernel,
            scalar_dim,
            [self.q, self.stage_left, self.normal_left, 2, direction],
        )
        self._launch(
            MNS.characteristic_to_conserved_kernel,
            scalar_dim,
            [self.q, self.stage_right, self.normal_right, 1, direction],
        )

    def _transverse_mlp(
        self,
        source_left: wp.array,
        source_right: wp.array,
        target_left: wp.array,
        target_right: wp.array,
        dim: tuple[int, int, int],
        lr: int,
        direction: int,
    ) -> None:
        scalar_dim = dim + (5,)
        params = self._mlp_kernel_inputs()
        transverse_kernel = MTS.transverse_mlp_kernel
        if self.mlp_precision == "float32":
            transverse_kernel = MTS32.transverse_mlp_f32_kernel
        self._launch(
            transverse_kernel,
            scalar_dim,
            [source_left, target_left, lr, direction] + params,
        )
        self._launch(
            transverse_kernel,
            scalar_dim,
            [source_right, target_right, lr, direction] + params,
        )

    def initialize(self) -> None:
        c = self.config
        self._launch(
            K.initialize_shockbubble_kernel,
            c.padded_shape[:3],
            [
                self.q,
                c.ghost,
                wp.float64(c.x_start),
                wp.float64(c.y_start),
                wp.float64(c.z_start),
                wp.float64(c.dx),
                wp.float64(c.dy),
                wp.float64(c.dz),
            ],
        )
        self._launch(K.conserved_to_primitive_kernel, c.padded_shape[:3], [self.q, self.primitive])

    def apply_boundary(self) -> None:
        c = self.config
        pz, py, px, _ = c.padded_shape
        self._launch(K.boundary_x_kernel, (pz, py), [self.q, c.nx])
        self._launch(K.boundary_y_kernel, (pz, px), [self.q, c.ny])
        self._launch(K.boundary_z_kernel, (py, px), [self.q, c.nz])

    def max_speed(self) -> float:
        c = self.config
        self.maximum.zero_()
        self._launch(K.max_speed_kernel, (c.nz, c.ny, c.nx), [self.primitive, self.maximum, c.ghost])
        return float(self.maximum.numpy()[0])

    def _x_fluxes(self, dt_dx: float) -> None:
        c = self.config
        pz, py, _, _ = c.padded_shape
        if self.mlp is None:
            self._launch(
                K.normal_x_kernel,
                (pz, py, c.nx + 2),
                [self.q, self.normal_left, self.normal_right],
            )
        else:
            self._normal_mlp((pz, py, c.nx + 2), 1)
        for flag in range(1, 5):
            first_lr = 1 if flag in (1, 3) else 2
            second_lr = 1 if flag in (1, 2) else 2
            if self.mlp is None:
                self._launch(
                    K.transverse_y_kernel,
                    (pz, c.ny, c.nx + 2),
                    [self.normal_left, self.normal_right, self.stage_left, self.stage_right, first_lr],
                )
                self._launch(
                    K.transverse_z_kernel,
                    (c.nz, c.ny, c.nx + 2),
                    [self.stage_left, self.stage_right, self.point_left, self.point_right, second_lr],
                )
            else:
                self._transverse_mlp(
                    self.normal_left, self.normal_right, self.stage_left, self.stage_right,
                    (pz, c.ny, c.nx + 2), first_lr, 2,
                )
                self._transverse_mlp(
                    self.stage_left, self.stage_right, self.point_left, self.point_right,
                    (c.nz, c.ny, c.nx + 2), second_lr, 3,
                )
            self._launch(
                K.flux_x_kernel,
                (c.nz, c.ny, c.nx + 1),
                [self.point_left, self.point_right, self.flux_x, wp.float64(dt_dx), flag],
            )

    def _y_fluxes(self, dt_dy: float) -> None:
        c = self.config
        pz, _, px, _ = c.padded_shape
        if self.mlp is None:
            self._launch(
                K.normal_y_kernel,
                (pz, c.ny + 2, px),
                [self.q, self.normal_left, self.normal_right],
            )
        else:
            self._normal_mlp((pz, c.ny + 2, px), 2)
        for flag in range(1, 5):
            first_lr = 1 if flag in (1, 3) else 2
            second_lr = 1 if flag in (1, 2) else 2
            if self.mlp is None:
                self._launch(
                    K.transverse_x_kernel,
                    (pz, c.ny + 2, c.nx),
                    [self.normal_left, self.normal_right, self.stage_left, self.stage_right, first_lr],
                )
                self._launch(
                    K.transverse_z_kernel,
                    (c.nz, c.ny + 2, c.nx),
                    [self.stage_left, self.stage_right, self.point_left, self.point_right, second_lr],
                )
            else:
                self._transverse_mlp(
                    self.normal_left, self.normal_right, self.stage_left, self.stage_right,
                    (pz, c.ny + 2, c.nx), first_lr, 1,
                )
                self._transverse_mlp(
                    self.stage_left, self.stage_right, self.point_left, self.point_right,
                    (c.nz, c.ny + 2, c.nx), second_lr, 3,
                )
            self._launch(
                K.flux_y_kernel,
                (c.nz, c.ny + 1, c.nx),
                [self.point_left, self.point_right, self.flux_y, wp.float64(dt_dy), flag],
            )

    def _z_fluxes(self, dt_dz: float) -> None:
        c = self.config
        _, py, px, _ = c.padded_shape
        if self.mlp is None:
            self._launch(
                K.normal_z_kernel,
                (c.nz + 2, py, px),
                [self.q, self.normal_left, self.normal_right],
            )
        else:
            self._normal_mlp((c.nz + 2, py, px), 3)
        for flag in range(1, 5):
            first_lr = 1 if flag in (1, 3) else 2
            second_lr = 1 if flag in (1, 2) else 2
            if self.mlp is None:
                self._launch(
                    K.transverse_y_kernel,
                    (c.nz + 2, c.ny, px),
                    [self.normal_left, self.normal_right, self.stage_left, self.stage_right, first_lr],
                )
                self._launch(
                    K.transverse_x_kernel,
                    (c.nz + 2, c.ny, c.nx),
                    [self.stage_left, self.stage_right, self.point_left, self.point_right, second_lr],
                )
            else:
                self._transverse_mlp(
                    self.normal_left, self.normal_right, self.stage_left, self.stage_right,
                    (c.nz + 2, c.ny, px), first_lr, 2,
                )
                self._transverse_mlp(
                    self.stage_left, self.stage_right, self.point_left, self.point_right,
                    (c.nz + 2, c.ny, c.nx), second_lr, 1,
                )
            self._launch(
                K.flux_z_kernel,
                (c.nz + 1, c.ny, c.nx),
                [self.point_left, self.point_right, self.flux_z, wp.float64(dt_dz), flag],
            )

    def _rk_stage(self, stage: int, dt: float) -> None:
        c = self.config
        self.apply_boundary()
        self._x_fluxes(dt / c.dx)
        self._y_fluxes(dt / c.dy)
        self._z_fluxes(dt / c.dz)
        dim = (c.nz, c.ny, c.nx)
        if stage == 1:
            self._launch(K.rk_stage1_kernel, dim, [self.q, self.q0, self.flux_x, self.flux_y, self.flux_z, c.ghost])
        elif stage == 2:
            self._launch(K.rk_stage2_kernel, dim, [self.q, self.q0, self.flux_x, self.flux_y, self.flux_z, c.ghost])
        else:
            self._launch(
                K.rk_stage3_kernel,
                dim,
                [self.q, self.q0, self.primitive, self.flux_x, self.flux_y, self.flux_z, c.ghost],
            )

    def advance(self) -> bool:
        c = self.config
        if self.time == c.t_end:
            return False
        maximum = self.max_speed()
        raw_dt = 1.0e10 if maximum < 1.0e-15 else c.cfl * min(c.dx, min(c.dy, c.dz)) / maximum
        dt = raw_dt
        if self.time + dt > c.t_end:
            dt = c.t_end - self.time
        start_time = self.time
        self.step += 1
        self.time += dt

        for stage in (1, 2, 3):
            self._rk_stage(stage, dt)

        self.dt_trace.append(
            {
                "step": self.step,
                "time_start": start_time,
                "raw_dt": raw_dt,
                "dt": dt,
                "time_end": self.time,
                "max_speed": maximum,
            }
        )
        return True

    def primitive_host(self) -> np.ndarray:
        c = self.config
        g = c.ghost
        values = self.primitive.numpy()
        return np.ascontiguousarray(values[g : g + c.nz, g : g + c.ny, g : g + c.nx, :])

    def diagnostics(self) -> dict[str, float | int]:
        values = self.primitive_host()
        return {
            "nan_count": int(np.count_nonzero(~np.isfinite(values))),
            "rho_min": float(np.min(values[..., 0])),
            "rho_max": float(np.max(values[..., 0])),
            "p_min": float(np.min(values[..., 4])),
            "p_max": float(np.max(values[..., 4])),
        }


def run_solver(
    config: ShockBubbleConfig,
    device: str,
    out_dir: str | Path,
    reference: str | Path | None = None,
    stop_step: int | None = None,
    strict_sync: bool = True,
    model: str | Path | None = None,
    eno_cutoff: bool = False,
) -> dict[str, object]:
    wp.init()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    solver = Weno5Rk3Solver(
        config,
        device=device,
        strict_sync=strict_sync,
        model=model,
        eno_cutoff=eno_cutoff,
    )
    wall_start = time.perf_counter()
    solver.initialize()
    initialized = solver.diagnostics()
    print(f"initialized device={solver.device} diagnostics={initialized}", flush=True)

    while solver.time != config.t_end and (stop_step is None or solver.step < stop_step):
        solver.advance()
        row = solver.dt_trace[-1]
        print(
            f"step={row['step']} raw_dt={row['raw_dt']:.17e} dt={row['dt']:.17e} "
            f"max_speed={row['max_speed']:.17e} t={row['time_end']:.17e}",
            flush=True,
        )

    wp.synchronize_device(solver.device)
    elapsed = time.perf_counter() - wall_start
    output_path = out_dir / f"step_{solver.step:04d}.bin"
    write_step(output_path, solver.time, solver.primitive_host())

    trace_path = out_dir / "dt_trace.csv"
    with trace_path.open("w", newline="", encoding="ascii") as stream:
        fieldnames = ("step", "time_start", "raw_dt", "dt", "time_end", "max_speed")
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(solver.dt_trace)

    report: dict[str, object] = {
        "config": asdict(config),
        "device": str(solver.device),
        "strict_sync": strict_sync,
        "reconstruction": (
            f"weno5-sr-v20-reflection-symmetric-mlp-{solver.mlp_precision}"
            if solver.mlp is not None
            else "weno5-js"
        ),
        "eno_cutoff": bool(eno_cutoff),
        "step": solver.step,
        "time": solver.time,
        "elapsed_seconds": elapsed,
        "output": str(output_path),
        "initial_diagnostics": initialized,
        "final_diagnostics": solver.diagnostics(),
    }
    if solver.mlp is not None:
        report["model"] = solver.mlp.manifest()
    if reference is not None and Path(reference).exists() and solver.time == config.t_end:
        comparison = compare_steps(output_path, reference)
        write_comparison(comparison, out_dir / "comparison.json")
        report["comparison"] = comparison
    (out_dir / "run_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii")
    return report
