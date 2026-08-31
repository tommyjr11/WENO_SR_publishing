from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from for_paper_results import config


GAMMA = 1.4


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=config.ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def environment_manifest() -> dict[str, Any]:
    try:
        import torch
        torch_info = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:  # pragma: no cover - diagnostic only
        torch_info = {"error": repr(exc)}
    try:
        import warp
        warp_version = getattr(warp, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - diagnostic only
        warp_version = f"unavailable: {exc!r}"
    return {
        "git_commit": git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch_info,
        "warp": warp_version,
        "models": {
            key: {
                "label": method.label,
                "path": str(method.model) if method.model else None,
                "sha256": sha256(method.model) if method.model else None,
                "mlp_precision": method.mlp_precision,
                "time_integrator": method.time_integrator,
            }
            for key, method in config.METHODS.items()
        },
    }


def primitive(conserved: np.ndarray, gamma: float = GAMMA) -> np.ndarray:
    rho = conserved[..., 0]
    safe_rho = np.maximum(rho, 1.0e-300)
    u = conserved[..., 1] / safe_rho
    v = conserved[..., 2] / safe_rho
    p = (gamma - 1.0) * (
        conserved[..., 3] - 0.5 * rho * (u * u + v * v)
    )
    return np.stack((rho, u, v, p), axis=-1)


def interior(state: np.ndarray, ghost: int, nx: int, ny: int) -> np.ndarray:
    return state[ghost : ghost + ny, ghost : ghost + nx, :]


def state_health(state: np.ndarray, ghost: int, nx: int, ny: int) -> dict[str, float | int | bool]:
    q = interior(state, ghost, nx, ny)
    pri = primitive(q)
    finite = np.isfinite(q).all() and np.isfinite(pri).all()
    return {
        "complete": bool(finite and np.min(pri[..., 0]) > 0.0 and np.min(pri[..., 3]) > 0.0),
        "nan_count": int(np.count_nonzero(~np.isfinite(q))),
        "rho_min": float(np.nanmin(pri[..., 0])),
        "rho_max": float(np.nanmax(pri[..., 0])),
        "p_min": float(np.nanmin(pri[..., 3])),
        "p_max": float(np.nanmax(pri[..., 3])),
    }


def rho_errors(numerical: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    delta = numerical - reference
    return {
        "rho_l1": float(np.mean(np.abs(delta))),
        "rho_l2": float(np.sqrt(np.mean(delta * delta))),
        "rho_linf": float(np.max(np.abs(delta))),
    }


def conservative_block_average(state: np.ndarray, target_n: int) -> np.ndarray:
    if state.shape[0] != state.shape[1] or state.shape[0] % target_n:
        raise ValueError(f"cannot block-average shape {state.shape} to {target_n}x{target_n}")
    factor = state.shape[0] // target_n
    return state.reshape(target_n, factor, target_n, factor, state.shape[-1]).mean(axis=(1, 3))

