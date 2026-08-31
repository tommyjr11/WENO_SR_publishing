#!/usr/bin/env python3
"""Compare classical and MLP WENO7/ADER4 on the 2D quadrant Riemann problem."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import warp_weno7_ader4_helpers_classical_only as wh
import warp_weno7_external_overwrite_only as ext
import weno7_ader4_warp_classical_only as weno7


wp = wh.wp


class TorchWeno7Beta:
    def __init__(self, model_path: Path, torch_device: str, gamma: float = 1.4):
        data = np.load(model_path, allow_pickle=True)
        expected = {
            "w1": (1, 6, 12),
            "b1": (1, 12),
            "w2": (1, 12, 8),
            "b2": (1, 8),
            "w3": (1, 8, 8),
            "b3": (1, 8),
            "w4": (1, 8, 4),
            "b4": (1, 4),
        }
        wrong = {k: data[k].shape for k, shape in expected.items() if k not in data.files or data[k].shape != shape}
        if wrong:
            raise ValueError(f"WENO7 model must be 6->12->8->8->4, got {wrong}")
        self.device = torch.device(torch_device)
        self.gamma = float(gamma)
        self.w1 = torch.as_tensor(data["w1"][0], dtype=torch.float64, device=self.device)
        self.b1 = torch.as_tensor(data["b1"][0], dtype=torch.float64, device=self.device)
        self.w2 = torch.as_tensor(data["w2"][0], dtype=torch.float64, device=self.device)
        self.b2 = torch.as_tensor(data["b2"][0], dtype=torch.float64, device=self.device)
        self.w3 = torch.as_tensor(data["w3"][0], dtype=torch.float64, device=self.device)
        self.b3 = torch.as_tensor(data["b3"][0], dtype=torch.float64, device=self.device)
        self.w4 = torch.as_tensor(data["w4"][0], dtype=torch.float64, device=self.device)
        self.b4 = torch.as_tensor(data["b4"][0], dtype=torch.float64, device=self.device)

    def _sync_after_copy(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @staticmethod
    def _swish(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def _mlp_beta(self, q: torch.Tensor) -> torch.Tensor:
        q0, q1, q2, q3, q4, q5, q6 = [q[..., k] for k in range(7)]
        d01 = -2.0 * q0 + 9.0 * q1 - 18.0 * q2 + 11.0 * q3
        d02 = -q0 + 4.0 * q1 - 5.0 * q2 + 2.0 * q3
        d03 = -q0 + 3.0 * q1 - 3.0 * q2 + q3
        d11 = q1 - 6.0 * q2 + 3.0 * q3 + 2.0 * q4
        d12 = q2 - 2.0 * q3 + q4
        d13 = -q1 + 3.0 * q2 - 3.0 * q3 + q4
        d21 = -2.0 * q2 - 3.0 * q3 + 6.0 * q4 - q5
        d22 = q2 - 2.0 * q3 + q4
        d23 = -q2 + 3.0 * q3 - 3.0 * q4 + q5
        d31 = -11.0 * q3 + 18.0 * q4 - 9.0 * q5 + 2.0 * q6
        d32 = 2.0 * q3 - 5.0 * q4 + 4.0 * q5 - q6
        d33 = -q3 + 3.0 * q4 - 3.0 * q5 + q6
        c1, c2, c3 = 1.0 / 36.0, 13.0 / 12.0, 781.0 / 720.0
        delta0 = c1 * d01.abs() + c2 * d02.abs() + c3 * d03.abs()
        delta1 = c1 * d11.abs() + c2 * d12.abs() + c3 * d13.abs()
        delta2 = c1 * d21.abs() + c2 * d22.abs() + c3 * d23.abs()
        delta3 = c1 * d31.abs() + c2 * d32.abs() + c3 * d33.abs()
        eps = 1.0e-15
        dd0 = q0 - 2.0 * q1 + q2
        dd1 = q1 - 2.0 * q2 + q3
        dd2 = q2 - 2.0 * q3 + q4
        dd3 = q3 - 2.0 * q4 + q5
        dd4 = q4 - 2.0 * q5 + q6
        g0 = dd0.abs() / ((q1 - q0).abs() + (q2 - q1).abs() + eps)
        g1 = dd1.abs() / ((q2 - q1).abs() + (q3 - q2).abs() + eps)
        g2 = dd2.abs() / ((q3 - q2).abs() + (q4 - q3).abs() + eps)
        g3 = dd3.abs() / ((q4 - q3).abs() + (q5 - q4).abs() + eps)
        g4 = dd4.abs() / ((q5 - q4).abs() + (q6 - q5).abs() + eps)
        gamma_s = torch.minimum(torch.ones_like(g0), torch.maximum(torch.maximum(torch.maximum(g0, g1), torch.maximum(g2, g3)), g4))
        delta_max = torch.maximum(torch.maximum(delta0, delta1), torch.maximum(delta2, delta3))
        inv_delta = 1.0 / torch.maximum(delta_max, torch.as_tensor(1.0e-15, dtype=torch.float64, device=q.device))
        q_scale = torch.maximum(q.abs().amax(dim=-1), torch.ones_like(delta_max))
        rel_scale = torch.maximum(delta_max / q_scale, torch.as_tensor(1.0e-30, dtype=torch.float64, device=q.device))
        scale_feature = torch.clamp((torch.log10(rel_scale) + 16.0) / 16.0, 0.0, 1.0)
        features = torch.stack([delta0 * inv_delta, delta1 * inv_delta, delta2 * inv_delta, delta3 * inv_delta, gamma_s, scale_feature], dim=-1)
        h1 = self._swish(features @ self.w1 + self.b1)
        h2 = self._swish(h1 @ self.w2 + self.b2)
        h3 = self._swish(h2 @ self.w3 + self.b3)
        raw = h3 @ self.w4 + self.b4
        beta = 4.0 * torch.softmax(6.0 * torch.tanh(raw / 6.0), dim=-1)
        plateau = delta_max <= 1.0e-13 * q_scale
        return torch.where(plateau[..., None], torch.ones_like(beta), beta)

    def _roe_average(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma
        tiny = 1.0e-15
        gm1 = gamma - 1.0
        rho_l = torch.clamp(left[..., 0], min=tiny)
        u_l = left[..., 1] / rho_l
        v_l = left[..., 2] / rho_l
        e_l = left[..., 3]
        p_l = torch.clamp(gm1 * (e_l - 0.5 * rho_l * (u_l * u_l + v_l * v_l)), min=tiny)
        h_l = (e_l + p_l) / rho_l
        rho_r = torch.clamp(right[..., 0], min=tiny)
        u_r = right[..., 1] / rho_r
        v_r = right[..., 2] / rho_r
        e_r = right[..., 3]
        p_r = torch.clamp(gm1 * (e_r - 0.5 * rho_r * (u_r * u_r + v_r * v_r)), min=tiny)
        h_r = (e_r + p_r) / rho_r
        sqrt_l = torch.sqrt(rho_l)
        sqrt_r = torch.sqrt(rho_r)
        inv = 1.0 / (sqrt_l + sqrt_r)
        u = (sqrt_l * u_l + sqrt_r * u_r) * inv
        v = (sqrt_l * v_l + sqrt_r * v_r) * inv
        h = (sqrt_l * h_l + sqrt_r * h_r) * inv
        rho = sqrt_l * sqrt_r
        q2 = u * u + v * v
        e = (rho * h + 0.5 * gm1 * rho * q2) / gamma
        return torch.stack([rho, rho * u, rho * v, e], dim=-1)

    def _jac_values(self, roe: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gamma = self.gamma
        tiny = 1.0e-15
        rho = torch.clamp(roe[..., 0], min=tiny)
        u = roe[..., 1] / rho
        v = roe[..., 2] / rho
        gm1 = gamma - 1.0
        q2 = u * u + v * v
        p = torch.clamp(gm1 * (roe[..., 3] - 0.5 * rho * q2), min=tiny)
        a = torch.sqrt(gamma * p / rho)
        h = 0.5 * q2 + a * a / gm1
        return u, v, a, h

    def _con_to_char(self, q: torch.Tensor, roe: torch.Tensor, direction: int) -> torch.Tensor:
        gamma = self.gamma
        u, v, a, h = self._jac_values(roe)
        gm1 = gamma - 1.0
        a2 = a * a
        scale = gm1 / (2.0 * a2)
        if direction == 1:
            c0 = ((h + a * (u - a) / gm1) * q[..., 0] - (u + a / gm1) * q[..., 1] - v * q[..., 2] + q[..., 3]) * scale
            c1 = ((-2.0 * h + 4.0 * a2 / gm1) * q[..., 0] + 2.0 * u * q[..., 1] + 2.0 * v * q[..., 2] - 2.0 * q[..., 3]) * scale
            c2 = (-2.0 * v * a2 / gm1 * q[..., 0] + 2.0 * a2 / gm1 * q[..., 2]) * scale
            c3 = ((h - a * (u + a) / gm1) * q[..., 0] + (-u + a / gm1) * q[..., 1] - v * q[..., 2] + q[..., 3]) * scale
        else:
            tangent = u
            normal = v
            c0 = ((h + a * (normal - a) / gm1) * q[..., 0] - tangent * q[..., 1] - (normal + a / gm1) * q[..., 2] + q[..., 3]) * scale
            c1 = ((-2.0 * h + 4.0 * a2 / gm1) * q[..., 0] + 2.0 * tangent * q[..., 1] + 2.0 * normal * q[..., 2] - 2.0 * q[..., 3]) * scale
            c2 = (-2.0 * tangent * a2 / gm1 * q[..., 0] + 2.0 * a2 / gm1 * q[..., 1]) * scale
            c3 = ((h - a * (normal + a) / gm1) * q[..., 0] - tangent * q[..., 1] + (-normal + a / gm1) * q[..., 2] + q[..., 3]) * scale
        return torch.stack([c0, c1, c2, c3], dim=-1)

    def _compute_beta(self, q: list[torch.Tensor], direction: int) -> torch.Tensor:
        shape = q[0].shape[:-1]
        beta = torch.empty(shape + (32,), dtype=torch.float64, device=self.device)
        for side, lr in ((0, 1), (1, 2)):
            roe = self._roe_average(q[3], q[4]) if lr == 1 else self._roe_average(q[2], q[3])
            chars = [self._con_to_char(qq, roe, direction) for qq in q]
            for comp in range(4):
                vals = torch.stack([c[..., comp] for c in chars], dim=-1)
                if lr == 1:
                    b = self._mlp_beta(vals)
                else:
                    b = torch.flip(self._mlp_beta(torch.flip(vals, dims=(-1,))), dims=(-1,))
                beta[..., side * 16 + comp * 4 : side * 16 + comp * 4 + 4] = b
        return beta

    @torch.no_grad()
    def fill(self, arrays: dict[str, object], params: wh.Params) -> None:
        u = wp.to_torch(arrays["u"])
        qx = [u[:, k : k + params.nx + 2, :] for k in range(7)]
        qy = [u[k : k + params.ny + 2, :, :] for k in range(7)]
        wp.to_torch(arrays["beta_x"]).copy_(self._compute_beta(qx, 1))
        wp.to_torch(arrays["beta_y"]).copy_(self._compute_beta(qy, 2))
        self._sync_after_copy()


def quadrant_primitive(x: np.ndarray, y: np.ndarray, quadrant_case: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rho = np.empty_like(x, dtype=np.float64)
    vx = np.empty_like(x, dtype=np.float64)
    vy = np.empty_like(x, dtype=np.float64)
    p = np.empty_like(x, dtype=np.float64)

    lower_left = (x < 0.5) & (y < 0.5)
    upper_left = (x < 0.5) & (y >= 0.5)
    lower_right = (x >= 0.5) & (y < 0.5)
    upper_right = (x >= 0.5) & (y >= 0.5)

    if quadrant_case == "case6":
        rho[lower_left], vx[lower_left], vy[lower_left], p[lower_left] = 0.8, 0.0, 0.0, 1.0
        rho[upper_left], vx[upper_left], vy[upper_left], p[upper_left] = 1.0, 0.7276, 0.0, 1.0
        rho[lower_right], vx[lower_right], vy[lower_right], p[lower_right] = 1.0, 0.0, 0.7276, 1.0
        rho[upper_right], vx[upper_right], vy[upper_right], p[upper_right] = 0.5315, 0.0, 0.0, 0.4
    else:
        rho[lower_left], vx[lower_left], vy[lower_left], p[lower_left] = 0.138, 1.206, 1.206, 0.029
        rho[upper_left], vx[upper_left], vy[upper_left], p[upper_left] = 0.5323, 1.206, 0.0, 0.3
        rho[lower_right], vx[lower_right], vy[lower_right], p[lower_right] = 0.5323, 0.0, 1.206, 0.3
        rho[upper_right], vx[upper_right], vy[upper_right], p[upper_right] = 1.5, 0.0, 0.0, 1.5
    return rho, vx, vy, p


def conserved_from_primitive_arrays(
    rho: np.ndarray,
    vx: np.ndarray,
    vy: np.ndarray,
    p: np.ndarray,
    gamma: float,
) -> np.ndarray:
    u = np.empty(rho.shape + (4,), dtype=np.float64)
    u[..., 0] = rho
    u[..., 1] = rho * vx
    u[..., 2] = rho * vy
    u[..., 3] = p / (gamma - 1.0) + 0.5 * rho * (vx * vx + vy * vy)
    return u


def make_quadrant_state(params: wh.Params, quadrant_case: str, quadrature: int) -> np.ndarray:
    if quadrature < 1:
        raise ValueError("--init-quadrature must be at least 1")
    xi, wi = np.polynomial.legendre.leggauss(quadrature)
    g = params.ghost
    jj, ii = np.indices((params.ny + 2 * g, params.nx + 2 * g))
    xc = (ii - g + 0.5) * params.dx
    yc = (jj - g + 0.5) * params.dy
    u = np.zeros((params.ny + 2 * g, params.nx + 2 * g, 4), dtype=np.float64)

    for sx, wx in zip(xi, wi):
        x = xc + 0.5 * params.dx * float(sx)
        for sy, wy in zip(xi, wi):
            y = yc + 0.5 * params.dy * float(sy)
            rho, vx, vy, p = quadrant_primitive(x, y, quadrant_case)
            u += 0.25 * float(wx) * float(wy) * conserved_from_primitive_arrays(rho, vx, vy, p, params.gamma)
    return u


def primitive_interior(u: np.ndarray, params: wh.Params) -> np.ndarray:
    g = params.ghost
    return wh.primitive_from_conserved(u[g : g + params.ny, g : g + params.nx, :], params.gamma)


def plot_qstyle(u: np.ndarray, params: wh.Params, out_path: Path, title: str) -> None:
    pri = primitive_interior(u, params)
    rho = pri[..., 0]
    vx = pri[..., 1]
    vy = pri[..., 2]
    p = pri[..., 3]

    x = np.linspace(0.0, 1.0, params.nx)
    y = np.linspace(0.0, 1.0, params.ny)
    x_grid, y_grid = np.meshgrid(x, y)
    rho_levels = np.arange(0.16, 1.71, 0.05)
    skip = max(1, params.nx // 30)

    fig, ax = plt.subplots(figsize=(6, 6))
    c = ax.contourf(x_grid, y_grid, p, levels=300, cmap="jet")
    ax.contour(x_grid, y_grid, rho, levels=rho_levels, colors="k", linewidths=0.3)
    ax.quiver(
        x_grid[::skip, ::skip],
        y_grid[::skip, ::skip],
        vx[::skip, ::skip],
        vy[::skip, ::skip],
        color="white",
        scale=40,
        width=0.002,
    )
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(c, ax=ax, label="Pressure")
    plt.tight_layout()
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def plot_difference(classical: np.ndarray, mlp: np.ndarray, params: wh.Params, out_path: Path) -> None:
    c_pri = primitive_interior(classical, params)
    m_pri = primitive_interior(mlp, params)
    rho_diff = m_pri[..., 0] - c_pri[..., 0]
    p_diff = m_pri[..., 3] - c_pri[..., 3]

    x = np.linspace(0.0, 1.0, params.nx)
    y = np.linspace(0.0, 1.0, params.ny)
    x_grid, y_grid = np.meshgrid(x, y)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    for ax, field, label in [
        (axes[0], rho_diff, r"$\rho_{\rm MLP}-\rho_{\rm classical}$"),
        (axes[1], p_diff, r"$p_{\rm MLP}-p_{\rm classical}$"),
    ]:
        scale = float(np.nanmax(np.abs(field)))
        if not np.isfinite(scale) or scale <= 0.0:
            scale = 1.0
        im = ax.contourf(x_grid, y_grid, field, levels=101, cmap="coolwarm", vmin=-scale, vmax=scale)
        ax.set_title(label)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(im, ax=ax)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_solution(
    initial: np.ndarray,
    params: wh.Params,
    args: argparse.Namespace,
    mlp_params: dict[str, object] | None,
    label: str,
) -> tuple[np.ndarray, dict[str, float]]:
    arrays = weno7.allocate_warp_arrays(initial.copy(), params, args.device)
    if mlp_params is not None:
        arrays.update(ext.allocate_external_beta_arrays(params.nx, params.ny, args.device))
    t = 0.0
    step = 0
    dt_values: list[float] = []
    print(f"{label}_start", flush=True)
    while t < args.t_end and step < args.max_steps:
        dt = wh.compute_dt_from_warp_array(arrays["u"], arrays["speed"], params, args.device)
        if t + dt > args.t_end:
            dt = args.t_end - t
        if dt <= 0.0:
            break
        if mlp_params is None:
            weno7.launch_weno7_ader4_step(
                arrays,
                params,
                dt,
                args.device,
                args.boundary,
                t,
                args.riemann_solver,
                None,
                False,
                "all",
            )
        else:
            boundary_kernel = wh.apply_periodic_boundary_kernel if args.boundary == "periodic" else wh.apply_boundary_kernel
            wp.launch(boundary_kernel, dim=(params.ny + 2 * params.ghost, params.nx + 2 * params.ghost), inputs=[arrays["u"], params.nx, params.ny, params.ghost], device=args.device)
            wp.synchronize()
            mlp_params.fill(arrays, params)
            ext.launch_weno7_ader4_step_external_beta_normal(
                arrays,
                params,
                dt,
                args.device,
                args.boundary,
                args.riemann_solver,
                args.eno_cutoff,
            )
        t += dt
        step += 1
        dt_values.append(float(dt))
        if args.report_interval > 0 and (step % args.report_interval == 0 or step == 1):
            host_now = arrays["u"].numpy()
            stats = wh.interior_stats(host_now, params)
            print(
                f"{label} step={step:04d} t={t:.8e} dt={dt:.8e} "
                f"rho=[{stats['rho_min']:.6e},{stats['rho_max']:.6e}] "
                f"p=[{stats['p_min']:.6e},{stats['p_max']:.6e}] nan={int(stats['nan_count'])}",
                flush=True,
            )
            if stats["nan_count"] or stats["rho_neg"] or stats["p_neg"]:
                print(f"{label}_failure: NaN/negative rho/p detected, stopping early", flush=True)
                break

    final = arrays["u"].numpy()
    stats = wh.interior_stats(final, params)
    summary = {
        "t": float(t),
        "steps": float(step),
        "dt_min": float(np.min(dt_values)) if dt_values else 0.0,
        "dt_max": float(np.max(dt_values)) if dt_values else 0.0,
        "dt_mean": float(np.mean(dt_values)) if dt_values else 0.0,
        **stats,
    }
    print(
        f"{label}_done steps={step} t={t:.8e} rho=[{stats['rho_min']:.6e},{stats['rho_max']:.6e}] "
        f"p=[{stats['p_min']:.6e},{stats['p_max']:.6e}] nan={int(stats['nan_count'])}",
        flush=True,
    )
    return final, summary


def diff_metrics(classical: np.ndarray, mlp: np.ndarray, params: wh.Params) -> dict[str, float]:
    c_pri = primitive_interior(classical, params)
    m_pri = primitive_interior(mlp, params)
    metrics: dict[str, float] = {}
    for name, idx in (("rho", 0), ("p", 3)):
        diff = m_pri[..., idx] - c_pri[..., idx]
        metrics[f"{name}_diff_l1"] = float(np.mean(np.abs(diff)))
        metrics[f"{name}_diff_l2"] = float(np.sqrt(np.mean(diff * diff)))
        metrics[f"{name}_diff_linf"] = float(np.max(np.abs(diff)))
    return metrics


def write_summary(path: Path, header: dict[str, object], sections: dict[str, dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for key, value in header.items():
            f.write(f"{key}: {value}\n")
        for name, values in sections.items():
            f.write(f"\n[{name}]\n")
            for key, value in values.items():
                f.write(f"{key}: {value}\n")


def run(args: argparse.Namespace) -> None:
    wh.require_warp()
    wp.init()
    wp.set_device(args.device)

    params = wh.Params(
        nx=args.nx,
        ny=args.ny,
        x_length=1.0,
        y_length=1.0,
        cfl=args.cfl,
        t_end=args.t_end,
    )
    if args.model is None and args.run_mlp:
        raise ValueError("--run-mlp requires --model")
    mlp_params = TorchWeno7Beta(args.model, args.device, params.gamma) if args.model is not None and args.run_mlp else None
    initial = make_quadrant_state(params, args.quadrant_case, args.init_quadrature)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"run_start solver={args.riemann_solver} nx={params.nx} ny={params.ny} cfl={params.cfl} "
        f"t_end={params.t_end} quadrant_case={args.quadrant_case} init_quadrature={args.init_quadrature} "
        f"run_classical={args.run_classical} run_mlp={args.run_mlp} model={args.model} eno_cutoff={args.eno_cutoff} "
        "external_beta_normal=torch-beta-overwrite",
        flush=True,
    )

    results: dict[str, np.ndarray] = {}
    summaries: dict[str, dict[str, float]] = {}
    if args.run_mlp:
        mlp, summaries["mlp"] = run_solution(initial, params, args, mlp_params, "mlp")
        results["mlp"] = mlp
        np.savez(args.out_dir / "mlp_quadrant_results.npz", initial=initial, mlp=mlp, **summaries["mlp"])
        plot_qstyle(
            mlp,
            params,
            args.out_dir / "mlp_pressure_rho_quiver_rho016_171_step005.png",
            f"WENO7 MLP {args.quadrant_case} {args.riemann_solver} {params.nx}x{params.ny} t={summaries['mlp']['t']:.3f}",
        )
    
    
    if args.run_classical:
        classical, summaries["classical"] = run_solution(initial, params, args, None, "classical")
        results["classical"] = classical
        np.savez(args.out_dir / "classical_quadrant_results.npz", initial=initial, classical=classical, **summaries["classical"])
        plot_qstyle(
            classical,
            params,
            args.out_dir / "classical_pressure_rho_quiver_rho016_171_step005.png",
            f"WENO7 classical {args.quadrant_case} {args.riemann_solver} {params.nx}x{params.ny} t={summaries['classical']['t']:.3f}",
        )

    if "classical" in results and "mlp" in results:
        summaries["mlp_minus_classical"] = diff_metrics(results["classical"], results["mlp"], params)
        plot_difference(results["classical"], results["mlp"], params, args.out_dir / "mlp_minus_classical_density_pressure.png")

    write_summary(
        args.out_dir / "summary.txt",
        {
            "model": str(args.model) if args.model is not None else "None",
            "quadrant_case": args.quadrant_case,
            "riemann_solver": args.riemann_solver,
            "nx": params.nx,
            "ny": params.ny,
            "cfl": params.cfl,
            "t_end": params.t_end,
            "init_quadrature": args.init_quadrature,
            "eno_cutoff": bool(args.eno_cutoff and args.run_mlp),
            "external_beta_normal": bool(args.run_mlp),
        },
        summaries,
    )
    print(f"summary={args.out_dir / 'summary.txt'}")
    if "classical" in results:
        print(f"classical_plot={args.out_dir / 'classical_pressure_rho_quiver_rho016_171_step005.png'}")
    if "mlp" in results:
        print(f"mlp_plot={args.out_dir / 'mlp_pressure_rho_quiver_rho016_171_step005.png'}")
    if "mlp_minus_classical" in summaries:
        print(f"diff_plot={args.out_dir / 'mlp_minus_classical_density_pressure.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--nx", type=int, default=400)
    parser.add_argument("--ny", type=int, default=400)
    parser.add_argument("--cfl", type=float, default=0.4)
    parser.add_argument("--t-end", type=float, default=0.5)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--weno-space", choices=("characteristic",), default="characteristic")
    parser.add_argument("--riemann-solver", choices=("evilin", "hllc"), default="evilin")
    parser.add_argument("--quadrant-case", choices=("case12", "case6"), default="case12")
    parser.add_argument("--boundary", choices=("outflow", "periodic"), default="outflow")
    parser.add_argument("--init-quadrature", type=int, default=5, help="tensor Gauss points per cell direction")
    parser.add_argument("--run-classical", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run-mlp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eno-cutoff", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--mlp-derivative-mode",
        choices=("normal",),
        default="normal",
        help="Compatibility option. This clean external runner only supports normal-direction MLP beta.",
    )
    parser.add_argument("--report-interval", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=10_000_000)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
