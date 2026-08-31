#!/usr/bin/env python3
"""Standalone WENO7/ADER4 Warp prototype matching ADER_TR4 HEOC."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import warp_weno7_ader4_helpers as wh

wp = wh.wp


if wp is not None:
    @wp.func
    def double_mach_exact_conserved(x: wp.float64, y: wp.float64, t: wp.float64, gamma: wp.float64) -> wp.vec4d:
        root3 = wp.sqrt(wp.float64(3.0))
        x0 = wp.float64(1.0) / wp.float64(6.0)
        shock_x = x0 + y / root3 + wp.float64(20.0) * t / root3
        if x < shock_x:
            return wh.pri_to_con(
                wp.vec4d(
                    wp.float64(8.0),
                    wp.float64(8.25) * root3 * wp.float64(0.5),
                    -wp.float64(8.25) * wp.float64(0.5),
                    wp.float64(116.5),
                ),
                gamma,
            )
        return wh.pri_to_con(wp.vec4d(wp.float64(1.4), wp.float64(0.0), wp.float64(0.0), wp.float64(1.0)), gamma)

    @wp.kernel
    def apply_double_mach_boundary_kernel(
        u: wp.array3d(dtype=wp.float64),
        nx: int,
        ny: int,
        gc: int,
        dx: wp.float64,
        dy: wp.float64,
        time: wp.float64,
        gamma: wp.float64,
    ):
        j, i = wp.tid()
        nx_total = nx + 2 * gc
        ny_total = ny + 2 * gc
        if i >= nx_total or j >= ny_total:
            return

        inside_x = i >= gc and i < nx + gc
        inside_y = j >= gc and j < ny + gc
        if inside_x and inside_y:
            return

        x = (wp.float64(i - gc) + wp.float64(0.5)) * dx
        y = (wp.float64(j - gc) + wp.float64(0.5)) * dy
        x0 = wp.float64(1.0) / wp.float64(6.0)

        q = wp.vec4d()
        use_exact = 0
        use_reflect = 0
        src_i = i
        src_j = j

        if j >= ny + gc:
            q = double_mach_exact_conserved(x, y, time, gamma)
            use_exact = 1
        elif j < gc:
            if x < x0:
                q = double_mach_exact_conserved(x, wp.float64(0.0), time, gamma)
                use_exact = 1
            else:
                use_reflect = 1
                src_j = 2 * gc - 1 - j
                if i < gc:
                    src_i = gc
                elif i >= nx + gc:
                    src_i = nx + gc - 1
        elif i < gc:
            q = double_mach_exact_conserved(x, y, time, gamma)
            use_exact = 1
        elif i >= nx + gc:
            src_i = nx + gc - 1
            src_j = j

        if use_exact == 1:
            for comp in range(4):
                u[j, i, comp] = q[comp]
        elif use_reflect == 1:
            u[j, i, 0] = u[src_j, src_i, 0]
            u[j, i, 1] = u[src_j, src_i, 1]
            u[j, i, 2] = -u[src_j, src_i, 2]
            u[j, i, 3] = u[src_j, src_i, 3]
        else:
            for comp in range(4):
                u[j, i, comp] = u[src_j, src_i, comp]


def allocate_warp_arrays(u0_host: np.ndarray, params: wh.Params, device: str) -> dict[str, object]:
    shape = params.padded_shape
    arrays = {
        "u": wp.array(u0_host, dtype=wp.float64, device=device),
        "pri": wp.zeros(shape, dtype=wp.float64, device=device),
        "flux_x": wp.zeros(shape, dtype=wp.float64, device=device),
        "flux_y": wp.zeros(shape, dtype=wp.float64, device=device),
        "speed": wp.zeros(params.nx * params.ny, dtype=wp.float64, device=device),
    }
    for name in ("l0", "r0", "l1", "r1", "l2", "r2", "l3", "r3", "tl1", "tl2", "tl3", "tr1", "tr2", "tr3"):
        arrays[name] = wp.zeros(shape, dtype=wp.float64, device=device)
    return arrays


def sod_initial_state(params: wh.Params, axis: str = "x", quadrature: int = 15) -> np.ndarray:
    if quadrature < 1:
        raise ValueError("--sod-init-quadrature must be at least 1")
    u = np.zeros(params.padded_shape, dtype=np.float64)
    gc = params.ghost
    if quadrature == 1:
        nodes = np.array([0.0], dtype=np.float64)
        weights = np.array([1.0], dtype=np.float64)
    else:
        nodes, weights = np.polynomial.legendre.leggauss(quadrature)
        weights = weights.astype(np.float64) / float(np.sum(weights))
        nodes = nodes.astype(np.float64)
    threshold = 0.5 * (params.x_length if axis == "x" else params.y_length)
    left = wh.primitive_to_conserved(1.0, 0.0, 0.0, 1.0, params.gamma)
    right = wh.primitive_to_conserved(0.125, 0.0, 0.0, 0.1, params.gamma)
    for j in range(params.ny + 2 * gc):
        for i in range(params.nx + 2 * gc):
            avg = np.zeros(4, dtype=np.float64)
            if axis == "x":
                center = (i - gc + 0.5) * params.dx
                half_width = 0.5 * params.dx
            else:
                center = (j - gc + 0.5) * params.dy
                half_width = 0.5 * params.dy
            for node, weight in zip(nodes, weights):
                coord = center + half_width * float(node)
                avg += float(weight) * (left if coord < threshold else right)
            u[j, i, :] = avg
    return u


def run_weno7_case(
    u0_host: np.ndarray,
    params: wh.Params,
    device: str,
    *,
    model_path: str | Path | None = None,
    mlp_params: dict[str, object] | None = None,
    eno_cutoff: bool = False,
    boundary: str = "outflow",
    riemann_solver: str = "evilin",
    mlp_derivative_mode: str = "all",
    max_steps: int = 10_000_000,
    report_interval: int = 0,
    label: str = "weno7",
) -> tuple[np.ndarray, dict[str, object]]:
    """Run the formal WENO7/ADER4 Warp time-advancement path."""
    wh.require_warp()
    wp.init()
    wp.set_device(device)
    if mlp_params is None and model_path is not None:
        mlp_params = load_mlp_params(model_path, device)
    arrays = allocate_warp_arrays(u0_host, params, device)
    t = 0.0
    step = 0
    dt_values: list[float] = []
    failed = False
    while step < max_steps:
        if params.t_end > 0.0 and t >= params.t_end:
            break
        dt = wh.compute_dt_from_warp_array(arrays["u"], arrays["speed"], params, device)
        if params.t_end > 0.0 and t + dt > params.t_end:
            dt = params.t_end - t
        if dt <= 0.0:
            failed = True
            break
        step += 1
        launch_weno7_ader4_step(
            arrays,
            params,
            dt,
            device,
            boundary,
            t,
            riemann_solver,
            mlp_params,
            eno_cutoff and mlp_params is not None,
            mlp_derivative_mode if mlp_params is not None else "all",
        )
        t += dt
        dt_values.append(float(dt))
        host_now = arrays["u"].numpy()
        stats_now = wh.interior_stats(host_now, params)
        if report_interval > 0 and (step == 1 or step % report_interval == 0):
            print(
                f"{label} step={step:04d} t={t:.8e} dt={dt:.8e} "
                f"rho=[{stats_now['rho_min']:.6e},{stats_now['rho_max']:.6e}] "
                f"p=[{stats_now['p_min']:.6e},{stats_now['p_max']:.6e}] "
                f"nan={int(stats_now['nan_count'])}",
                flush=True,
            )
        if stats_now["nan_count"] > 0 or stats_now["rho_neg"] > 0 or stats_now["p_neg"] > 0:
            failed = True
            break
    final_host = arrays["u"].numpy()
    stats = wh.interior_stats(final_host, params)
    if stats["nan_count"] > 0 or stats["rho_neg"] > 0 or stats["p_neg"] > 0:
        failed = True
    summary: dict[str, object] = {
        "t": float(t),
        "steps": int(step),
        "failed": bool(failed),
        "dt_values": dt_values,
        "dt_min": float(np.min(dt_values)) if dt_values else 0.0,
        "dt_max": float(np.max(dt_values)) if dt_values else 0.0,
        "dt_mean": float(np.mean(dt_values)) if dt_values else 0.0,
        **stats,
    }
    print(
        f"{label}_done steps={step} t={t:.8e} failed={int(failed)} "
        f"rho=[{stats['rho_min']:.6e},{stats['rho_max']:.6e}] "
        f"p=[{stats['p_min']:.6e},{stats['p_max']:.6e}] "
        f"nan={int(stats['nan_count'])}",
        flush=True,
    )
    return final_host, summary


def run_sod_case(
    params: wh.Params,
    device: str,
    *,
    axis: str = "x",
    init_quadrature: int = 15,
    model_path: str | Path | None = None,
    eno_cutoff: bool = False,
    riemann_solver: str = "evilin",
    mlp_derivative_mode: str = "all",
    max_steps: int = 1_000_000,
    report_interval: int = 0,
    label: str = "sod",
) -> tuple[np.ndarray, dict[str, object]]:
    return run_weno7_case(
        sod_initial_state(params, axis, init_quadrature),
        params,
        device,
        model_path=model_path,
        eno_cutoff=eno_cutoff,
        boundary="outflow",
        riemann_solver=riemann_solver,
        mlp_derivative_mode=mlp_derivative_mode,
        max_steps=max_steps,
        report_interval=report_interval,
        label=label,
    )


def load_mlp_params(model_path: str | Path, device: str) -> dict[str, object]:
    data = np.load(model_path, allow_pickle=True)
    required = ("w1", "b1", "w2", "b2", "w3", "b3", "w4", "b4")
    missing = [name for name in required if name not in data.files]
    if missing:
        raise ValueError(f"Model {model_path} is missing arrays: {missing}")
    expected_shapes = {
        "w1": (1, 6, 12),
        "b1": (1, 12),
        "w2": (1, 12, 8),
        "b2": (1, 8),
        "w3": (1, 8, 8),
        "b3": (1, 8),
        "w4": (1, 8, 4),
        "b4": (1, 4),
    }
    wrong = {name: data[name].shape for name, shape in expected_shapes.items() if data[name].shape != shape}
    if wrong:
        raise ValueError(
            f"Model {model_path} uses incompatible MLP shapes: {wrong}. "
            "weno7_ader4_warp.py only supports the WENO7 6->12->8->8->4 checkpoints."
        )
    if "meta_json" in data.files:
        meta = str(data["meta_json"])
        if "shared_direct_beta_ratio_6_12_8_8_4" not in meta:
            print("warning: model metadata does not mention shared_direct_beta_ratio_6_12_8_8_4")
    return {name: wp.array(data[name], dtype=wp.float64, device=device, requires_grad=False) for name in required}


def launch_weno7_ader4_step(
    arrays: dict[str, object],
    params: wh.Params,
    dt: float,
    device: str,
    boundary: str,
    boundary_time: float = 0.0,
    riemann_solver: str = "evilin",
    mlp_params: dict[str, object] | None = None,
    eno_cutoff: bool = False,
    mlp_derivative_mode: str = "all",
) -> None:
    if riemann_solver not in ("evilin", "hllc"):
        raise ValueError("weno7_ader4_warp.py supports --riemann-solver evilin or hllc")
    if mlp_derivative_mode not in ("all", "classical", "normal"):
        raise ValueError("--mlp-derivative-mode must be 'all', 'classical', or 'normal'")
    nx = params.nx
    ny = params.ny
    gc = params.ghost
    nx_total = nx + 2 * gc
    ny_total = ny + 2 * gc
    tempdx_dt = dt / params.dx
    tempdy_dt = dt / params.dy
    solver_kind = 1 if riemann_solver == "hllc" else 0
    eno_cutoff_i = 1 if eno_cutoff and mlp_params is not None else 0
    mlp_value_only_i = 1 if mlp_params is not None and mlp_derivative_mode in ("classical", "normal") else 0
    mlp_cross_i = 1 if mlp_params is not None and mlp_derivative_mode != "normal" else 0

    if boundary == "double-mach":
        wp.launch(
            apply_double_mach_boundary_kernel,
            dim=(ny_total, nx_total),
            inputs=[
                arrays["u"],
                nx,
                ny,
                gc,
                wp.float64(params.dx),
                wp.float64(params.dy),
                wp.float64(boundary_time),
                wp.float64(params.gamma),
            ],
            device=device,
        )
    else:
        boundary_kernel = wh.apply_periodic_boundary_kernel if boundary == "periodic" else wh.apply_boundary_kernel
        wp.launch(boundary_kernel, dim=(ny_total, nx_total), inputs=[arrays["u"], nx, ny, gc], device=device)
    if mlp_params is None:
        wp.launch(
            wh.compute_x_stage_weno7_big_kernel,
            dim=(ny + 8, nx + 2),
            inputs=[arrays["u"], arrays["l0"], arrays["r0"], arrays["l1"], arrays["r1"], arrays["l2"], arrays["r2"], arrays["l3"], arrays["r3"], nx, ny, wp.float64(params.dx), wp.float64(params.gamma)],
            device=device,
        )
    else:
        wp.launch(
            wh.compute_x_stage_weno7_mlp_big_kernel,
            dim=(ny + 8, nx + 2),
            inputs=[
                arrays["u"], arrays["l0"], arrays["r0"], arrays["l1"], arrays["r1"], arrays["l2"], arrays["r2"], arrays["l3"], arrays["r3"],
                mlp_params["w1"], mlp_params["b1"], mlp_params["w2"], mlp_params["b2"], mlp_params["w3"], mlp_params["b3"], mlp_params["w4"], mlp_params["b4"],
                nx, ny, wp.float64(params.dx), wp.float64(params.gamma), eno_cutoff_i, mlp_value_only_i,
            ],
            device=device,
        )
    for loca in (1, 2):
        if mlp_params is None or mlp_cross_i == 0:
            wp.launch(
                wh.compute_x_cross_stage_ader4_kernel,
                dim=(ny, nx + 2),
                inputs=[arrays["tl1"], arrays["tl2"], arrays["tl3"], arrays["tr1"], arrays["tr2"], arrays["tr3"], arrays["l0"], arrays["r0"], arrays["l1"], arrays["r1"], arrays["l2"], arrays["r2"], arrays["l3"], arrays["r3"], nx, ny, wp.float64(params.dy), wp.float64(dt), loca, wp.float64(params.gamma)],
                device=device,
            )
        else:
            wp.launch(
                wh.compute_x_cross_stage_ader4_mlp_kernel,
                dim=(ny, nx + 2),
                inputs=[
                    arrays["tl1"], arrays["tl2"], arrays["tl3"], arrays["tr1"], arrays["tr2"], arrays["tr3"],
                    arrays["l0"], arrays["r0"], arrays["l1"], arrays["r1"], arrays["l2"], arrays["r2"], arrays["l3"], arrays["r3"],
                    mlp_params["w1"], mlp_params["b1"], mlp_params["w2"], mlp_params["b2"], mlp_params["w3"], mlp_params["b3"], mlp_params["w4"], mlp_params["b4"],
                    nx, ny, wp.float64(params.dy), wp.float64(dt), loca, wp.float64(params.gamma), eno_cutoff_i, mlp_value_only_i,
                ],
                device=device,
            )
        wp.launch(
            wh.compute_x_flux_ader4_kernel,
            dim=(ny, nx + 1),
            inputs=[arrays["flux_x"], arrays["tl1"], arrays["tl2"], arrays["tl3"], arrays["tr1"], arrays["tr2"], arrays["tr3"], wp.float64(tempdx_dt), nx, ny, loca, wp.float64(params.gamma), solver_kind],
            device=device,
        )

    if mlp_params is None:
        wp.launch(
            wh.compute_y_stage_weno7_big_kernel,
            dim=(ny + 2, nx + 8),
            inputs=[arrays["u"], arrays["l0"], arrays["r0"], arrays["l1"], arrays["r1"], arrays["l2"], arrays["r2"], arrays["l3"], arrays["r3"], nx, ny, wp.float64(params.dy), wp.float64(params.gamma)],
            device=device,
        )
    else:
        wp.launch(
            wh.compute_y_stage_weno7_mlp_big_kernel,
            dim=(ny + 2, nx + 8),
            inputs=[
                arrays["u"], arrays["l0"], arrays["r0"], arrays["l1"], arrays["r1"], arrays["l2"], arrays["r2"], arrays["l3"], arrays["r3"],
                mlp_params["w1"], mlp_params["b1"], mlp_params["w2"], mlp_params["b2"], mlp_params["w3"], mlp_params["b3"], mlp_params["w4"], mlp_params["b4"],
                nx, ny, wp.float64(params.dy), wp.float64(params.gamma), eno_cutoff_i, mlp_value_only_i,
            ],
            device=device,
        )
    for loca in (1, 2):
        if mlp_params is None or mlp_cross_i == 0:
            wp.launch(
                wh.compute_y_cross_stage_ader4_kernel,
                dim=(ny + 2, nx),
                inputs=[arrays["tl1"], arrays["tl2"], arrays["tl3"], arrays["tr1"], arrays["tr2"], arrays["tr3"], arrays["l0"], arrays["r0"], arrays["l1"], arrays["r1"], arrays["l2"], arrays["r2"], arrays["l3"], arrays["r3"], nx, ny, wp.float64(params.dx), wp.float64(dt), loca, wp.float64(params.gamma)],
                device=device,
            )
        else:
            wp.launch(
                wh.compute_y_cross_stage_ader4_mlp_kernel,
                dim=(ny + 2, nx),
                inputs=[
                    arrays["tl1"], arrays["tl2"], arrays["tl3"], arrays["tr1"], arrays["tr2"], arrays["tr3"],
                    arrays["l0"], arrays["r0"], arrays["l1"], arrays["r1"], arrays["l2"], arrays["r2"], arrays["l3"], arrays["r3"],
                    mlp_params["w1"], mlp_params["b1"], mlp_params["w2"], mlp_params["b2"], mlp_params["w3"], mlp_params["b3"], mlp_params["w4"], mlp_params["b4"],
                    nx, ny, wp.float64(params.dx), wp.float64(dt), loca, wp.float64(params.gamma), eno_cutoff_i, mlp_value_only_i,
                ],
                device=device,
            )
        wp.launch(
            wh.compute_y_flux_ader4_kernel,
            dim=(ny + 1, nx),
            inputs=[arrays["flux_y"], arrays["tl1"], arrays["tl2"], arrays["tl3"], arrays["tr1"], arrays["tr2"], arrays["tr3"], wp.float64(tempdy_dt), nx, ny, loca, wp.float64(params.gamma), solver_kind],
            device=device,
        )

    wp.launch(
        wh.update_ader4_kernel,
        dim=(ny, nx),
        inputs=[arrays["u"], arrays["flux_x"], arrays["flux_y"], arrays["pri"], nx, ny, gc, wp.float64(params.gamma)],
        device=device,
    )
    wp.synchronize()


def read_cuda_csv_fields(path: str | Path, params: wh.Params) -> tuple[float | None, np.ndarray]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8") as file:
        header = file.readline().strip()
        rows = [line.strip() for line in file if line.strip()]
    time = None
    if header.startswith("# Time:"):
        time = float(header.split(":", 1)[1].strip())
    ny = len(rows)
    data = np.fromstring("\n".join(rows), sep=",", dtype=np.float64)
    if ny == 0 or data.size % (ny * 4) != 0:
        raise ValueError(f"{csv_path} has malformed CSV dimensions")
    nx = data.size // (ny * 4)
    fields = data.reshape(ny, nx, 4)
    if (ny, nx) != (params.ny, params.nx):
        raise ValueError(f"{csv_path} shape is {(ny, nx)}, but Warp params expect {(params.ny, params.nx)}")
    return time, fields


def density_comparison(warp_u: np.ndarray, csv_path: str | Path, params: wh.Params) -> dict[str, float]:
    csv_time, cuda_pri = read_cuda_csv_fields(csv_path, params)
    gc = params.ghost
    warp_rho = warp_u[gc : gc + params.ny, gc : gc + params.nx, 0]
    cuda_rho = cuda_pri[..., 0]
    diff = warp_rho - cuda_rho
    abs_diff = np.abs(diff)
    max_flat = int(np.argmax(abs_diff))
    max_j, max_i = np.unravel_index(max_flat, abs_diff.shape)
    eps = 1.0e-300
    return {
        "csv_time": float("nan") if csv_time is None else csv_time,
        "mean_abs": float(np.mean(abs_diff)),
        "rms_abs": float(np.sqrt(np.mean(diff * diff))),
        "linf_abs": float(np.max(abs_diff)),
        "rel_l1": float(np.sum(abs_diff) / max(np.sum(np.abs(cuda_rho)), eps)),
        "rel_l2": float(np.linalg.norm(diff.ravel()) / max(np.linalg.norm(cuda_rho.ravel()), eps)),
        "rel_linf": float(np.max(abs_diff) / max(np.max(np.abs(cuda_rho)), eps)),
        "max_i": float(max_i),
        "max_j": float(max_j),
        "max_x": float((max_i + 0.5) * params.dx),
        "max_y": float((max_j + 0.5) * params.dy),
        "warp_rho_at_max": float(warp_rho[max_j, max_i]),
        "cuda_rho_at_max": float(cuda_rho[max_j, max_i]),
    }


def print_density_comparison(metrics: dict[str, float], csv_path: str | Path) -> None:
    print(f"compare_csv={csv_path}")
    if not np.isnan(metrics["csv_time"]):
        print(f"csv_time={metrics['csv_time']:.16e}")
    print(
        "density_diff: "
        f"mean_abs={metrics['mean_abs']:.16e} "
        f"rms_abs={metrics['rms_abs']:.16e} "
        f"linf_abs={metrics['linf_abs']:.16e}"
    )
    print(
        "density_rel: "
        f"rel_l1={metrics['rel_l1']:.16e} "
        f"rel_l2={metrics['rel_l2']:.16e} "
        f"rel_linf={metrics['rel_linf']:.16e}"
    )
    print(
        "density_max_location: "
        f"i={int(metrics['max_i'])} j={int(metrics['max_j'])} "
        f"x={metrics['max_x']:.16e} y={metrics['max_y']:.16e} "
        f"warp_rho={metrics['warp_rho_at_max']:.16e} "
        f"cuda_rho={metrics['cuda_rho_at_max']:.16e}"
    )


def run_demo(args: argparse.Namespace) -> None:
    wh.require_warp()
    wp = wh.wp
    wp.init()
    params = wh.Params(nx=args.nx, ny=args.ny, x_length=args.x_length, y_length=args.y_length, cfl=args.cfl, t_end=args.t_end)
    wp.set_device(args.device)
    mlp_params = load_mlp_params(args.model, args.device) if args.model else None

    if args.initial_condition == "shock-bubble":
        u0_host = wh.make_initial_state(params)
    else:
        u0_host = sod_initial_state(params, "x", args.sod_init_quadrature)
    arrays = allocate_warp_arrays(u0_host, params, args.device)
    initial_stats = wh.interior_stats(u0_host, params)
    print(
        f"start: scheme={'WENO7_ADER4_MLP' if mlp_params is not None else 'WENO7_ADER4'} "
        f"device={args.device} initial_condition={args.initial_condition} boundary={args.boundary} "
        f"riemann_solver={args.riemann_solver} mass={initial_stats['mass']:.16e}, "
        f"rho=[{initial_stats['rho_min']:.6e},{initial_stats['rho_max']:.6e}], "
        f"p=[{initial_stats['p_min']:.6e},{initial_stats['p_max']:.6e}] "
        f"model={args.model or 'classical'} eno_cutoff={bool(args.eno_cutoff and mlp_params is not None)} "
        f"mlp_derivative_mode={args.mlp_derivative_mode if mlp_params is not None else 'classical'}"
    )

    t = 0.0
    step = 0
    max_steps = args.steps if args.steps > 0 else 10_000_000
    while step < max_steps:
        dt = wh.compute_dt_from_warp_array(arrays["u"], arrays["speed"], params, args.device)
        if args.t_end > 0.0 and t + dt > args.t_end:
            dt = args.t_end - t
        if dt <= 0.0:
            break
        step += 1
        launch_weno7_ader4_step(
            arrays,
            params,
            dt,
            args.device,
            args.boundary,
            t,
            args.riemann_solver,
            mlp_params,
            args.eno_cutoff,
            args.mlp_derivative_mode,
        )
        t += dt
        if args.report_interval > 0 and (step % args.report_interval == 0 or step == 1):
            host = arrays["u"].numpy()
            stats = wh.interior_stats(host, params)
            print(
                f"step={step} t={t:.16e} dt={dt:.16e} mass={stats['mass']:.16e} "
                f"rho=[{stats['rho_min']:.6e},{stats['rho_max']:.6e}] "
                f"p=[{stats['p_min']:.6e},{stats['p_max']:.6e}] nan={int(stats['nan_count'])}"
            )
        if args.t_end > 0.0 and t >= args.t_end:
            break

    final_host = arrays["u"].numpy()
    stats = wh.interior_stats(final_host, params)
    print(
        f"done: steps={step} t={t:.16e} mass={stats['mass']:.16e} "
        f"rho=[{stats['rho_min']:.6e},{stats['rho_max']:.6e}] "
        f"p=[{stats['p_min']:.6e},{stats['p_max']:.6e}] nan={int(stats['nan_count'])} "
        f"rho_neg={int(stats['rho_neg'])} p_neg={int(stats['p_neg'])}"
    )

    if args.save_npy:
        np.save(args.save_npy, final_host)
        print(f"saved_npy={args.save_npy}")
    if args.compare_csv:
        metrics = density_comparison(final_host, args.compare_csv, params)
        print_density_comparison(metrics, args.compare_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=450)
    parser.add_argument("--ny", type=int, default=178)
    parser.add_argument("--x-length", type=float, default=0.225)
    parser.add_argument("--y-length", type=float, default=0.089)
    parser.add_argument("--cfl", type=float, default=0.228)
    parser.add_argument("--t-end", type=float, default=0.0002)
    parser.add_argument("--steps", type=int, default=0, help="fixed number of steps; 0 means run to --t-end")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--compare-csv", type=str, default=None)
    parser.add_argument("--save-npy", type=str, default=None)
    parser.add_argument("--model", type=str, default=None, help="optional WENO7 6->12->8->8->4 MLP checkpoint")
    parser.add_argument("--eno-cutoff", action=argparse.BooleanOptionalAction, default=False, help="apply inference-only MLP ENO cutoff")
    parser.add_argument("--sod-init-quadrature", type=int, default=15, help="Gauss points for Sod conserved cell-average initialization")
    parser.add_argument(
        "--mlp-derivative-mode",
        choices=("all", "classical", "normal"),
        default="all",
        help=(
            "all: MLP weights for value, derivatives, and cross-stage reconstructions; "
            "classical: MLP for state values including cross-stage values, classical JS beta for derivative quantities; "
            "normal: MLP only for normal-direction characteristic l0/r0 face values, all derivatives and cross-stage reconstructions classical"
        ),
    )
    parser.add_argument("--report-interval", type=int, default=100)
    parser.add_argument("--boundary", choices=("outflow", "periodic", "double-mach"), default="outflow", help="ghost-cell boundary condition")
    parser.add_argument("--initial-condition", choices=("shock-bubble", "sod-x"), default="shock-bubble")
    parser.add_argument("--riemann-solver", choices=("evilin", "hllc"), default="evilin", help="Riemann solver")
    return parser.parse_args()


if __name__ == "__main__":
    run_demo(parse_args())
