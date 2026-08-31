#!/usr/bin/env python3
"""Verify the WENO-SR publication snapshot without running formal CFD jobs."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
SNAPSHOT = REPOSITORY / "code" / "snapshot"
MAX_BYTES = 50 * 1024 * 1024
sys.dont_write_bytecode = True

MODELS = {
    "weno5_sr_fp64_step012250.npz": {
        "sha256": "368759415a7dcf5567af76db06fd49c0b89260150f0fab7d9ada7f067ab3f74e",
        "dtype": "float64",
        "shapes": {
            "w1": (1, 5, 10), "b1": (1, 10),
            "w2": (1, 10, 6), "b2": (1, 6),
            "w3": (1, 6, 6), "b3": (1, 6),
            "w4": (1, 6, 3), "b4": (1, 3),
        },
        "step": 12250,
        "reflection": "0.5*(M(x)+P*M(P*x))",
    },
    "weno5_sr_fp32_step016500.npz": {
        "sha256": "c88441a950b91713353685edc0aa4debcb848fdddb1ba1b9442dd893a40600bc",
        "dtype": "float32",
        "shapes": {
            "w1": (1, 5, 10), "b1": (1, 10),
            "w2": (1, 10, 6), "b2": (1, 6),
            "w3": (1, 6, 6), "b3": (1, 6),
            "w4": (1, 6, 3), "b4": (1, 3),
        },
        "step": 16500,
        "reflection": "0.5*(M(x)+P*M(P*x))",
    },
    "weno7_sr_fp64_step016750.npz": {
        "sha256": "0a55fd07a87e73b28e1c471991322dc256ddebccdbfdf2d5ba3722ae8dde3d93",
        "dtype": "float64",
        "shapes": {
            "w1": (1, 6, 24), "b1": (1, 24),
            "w2": (1, 24, 16), "b2": (1, 16),
            "w3": (1, 16, 16), "b3": (1, 16),
            "w4": (1, 16, 4), "b4": (1, 4),
        },
        "step": 16750,
        "reflection": "0.5*(M(x)+P4*M(P6*x))",
    },
}

MODEL_ALIASES = {
    "weno5_sr_fp64_step012250.npz": (
        "code/snapshot/teacherfree_lab_weno5_v20_distance_balanced/runs/"
        "apost_weno5_v20_distance_balanced_cfl05_200k/checkpoints/"
        "model_step_012250.npz"
    ),
    "weno5_sr_fp32_step016500.npz": (
        "code/snapshot/teacherfree_lab_weno5_v20_distance_balanced_mlp_f32_fast/runs/"
        "apost_weno5_v20_mlp_f32_fast_200k/checkpoints/model_step_016500.npz"
    ),
    "weno7_sr_fp64_step016750.npz": (
        "code/snapshot/teacherfree_lab_weno7_rk4_distance_balanced_fast/runs/"
        "apost_weno7_rk4_distance_balanced_fast_4090_200k/checkpoints/"
        "model_step_016750.npz"
    ),
}

EXPERIMENTS = (
    "01_gste_long_time_advection",
    "02_sod_shock_tube",
    "03_lax_shock_tube",
    "04_titarev_toro",
    "05_isentropic_vortex",
    "06_riemann_c3",
    "07_riemann_c4",
    "08_riemann_c5",
    "09_riemann_c6",
    "10_double_mach_reflection",
    "11_shock_bubble_2d_ma122",
    "12_shock_bubble_2d_ma30",
    "13_shock_bubble_3d_ma30",
    "14_mixed_precision_timing",
)

FORBIDDEN_SUFFIXES = {".npy", ".bin", ".pt", ".pth"}
FORBIDDEN_PARTS = {"raw", "logs", "__pycache__", ".pytest_cache"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def release_files() -> list[Path]:
    return sorted(
        path
        for path in REPOSITORY.rglob("*")
        if path.is_file()
        and ".git" not in path.relative_to(REPOSITORY).parts
        and "__pycache__" not in path.relative_to(REPOSITORY).parts
        and path.name != "MANIFEST.sha256"
    )


def verify_models() -> list[str]:
    messages: list[str] = []
    for name, expected in MODELS.items():
        path = REPOSITORY / "models" / name
        if not path.is_file():
            raise AssertionError(f"missing selected checkpoint: {path}")
        digest = sha256(path)
        if digest != expected["sha256"]:
            raise AssertionError(f"checkpoint hash mismatch: {name}: {digest}")
        with np.load(path, allow_pickle=False) as data:
            for key, shape in expected["shapes"].items():
                if key not in data.files:
                    raise AssertionError(f"{name} is missing {key}")
                if data[key].shape != shape:
                    raise AssertionError(f"{name}:{key} shape {data[key].shape} != {shape}")
                if str(data[key].dtype) != expected["dtype"]:
                    raise AssertionError(
                        f"{name}:{key} dtype {data[key].dtype} != {expected['dtype']}"
                    )
            metadata = json.loads(str(data["meta_json"].item()))
        if metadata.get("raw_step") != expected["step"]:
            raise AssertionError(f"{name} step metadata mismatch")
        if metadata.get("reflection_formula") != expected["reflection"]:
            raise AssertionError(f"{name} reflection metadata mismatch")
        alias = REPOSITORY / MODEL_ALIASES[name]
        if not alias.is_file() or sha256(alias) != digest:
            raise AssertionError(f"runtime checkpoint alias mismatch: {alias}")
        messages.append(f"model ok: {name} ({expected['dtype']}, step {expected['step']})")
    return messages


def verify_reflection() -> list[str]:
    try:
        import torch
    except ImportError as exc:
        raise AssertionError("PyTorch is required for reflection verification") from exc

    sys.path.insert(0, str(SNAPSHOT))
    features5 = torch.tensor(
        [[0.12, 0.67, 1.0, 0.31, 0.82], [1.0, 0.28, 0.04, 0.73, 0.46]],
        dtype=torch.float64,
    )
    features7 = torch.tensor(
        [
            [0.09, 0.36, 0.81, 1.0, 0.27, 0.73],
            [1.0, 0.71, 0.22, 0.03, 0.64, 0.41],
        ],
        dtype=torch.float64,
    )

    v12 = importlib.import_module("teacherfree_lab_weno5_v12_reflection_sym.v12_model")
    model5 = v12.load_checkpoint(REPOSITORY / "models/weno5_sr_fp64_step012250.npz", "cpu").eval()
    defect5 = v12.reflection_defect(model5, features5)

    fast_dir = SNAPSHOT / "teacherfree_lab_weno5_v20_distance_balanced_mlp_f32_fast"
    sys.path.insert(0, str(fast_dir))
    try:
        mixed = importlib.import_module("model")
        model5f32 = mixed.load_checkpoint(
            REPOSITORY / "models/weno5_sr_fp32_step016500.npz", "cpu"
        ).eval()
        defect5f32 = mixed.reflection_defect(model5f32, features5)
    finally:
        sys.path.remove(str(fast_dir))

    w7 = importlib.import_module("teacherfree_lab_weno7_rk4_distance_balanced_fast.weno7_core")
    model7 = w7.load_checkpoint(REPOSITORY / "models/weno7_sr_fp64_step016750.npz", "cpu").eval()
    defect7 = w7.reflection_defect(model7, features7)

    with torch.no_grad():
        outputs = {
            "WENO5 FP64": model5(features5).detach().cpu().numpy(),
            "WENO5 FP32": model5f32(features5).detach().cpu().double().numpy(),
            "WENO7 FP64": model7(features7).detach().cpu().numpy(),
        }
    expected_outputs = {
        "WENO5 FP64": np.array(
            [
                [0.09531682872989035, 0.33537291374035355, 0.5693102575297561],
                [0.5541032210177954, 0.33931186577461137, 0.10658491320759339],
            ]
        ),
        "WENO5 FP32": np.array(
            [
                [0.09437001636251807, 0.3250548988580704, 0.5805750489234924],
                [0.5602236539125443, 0.33178114891052246, 0.1079952244181186],
            ]
        ),
        "WENO7 FP64": np.array(
            [
                [
                    0.04199914125039414,
                    0.1136805066940093,
                    0.2572032421753548,
                    0.5871171098802417,
                ],
                [
                    0.6620590842915794,
                    0.22888495913119702,
                    0.0859246392218968,
                    0.02313131735532667,
                ],
            ]
        ),
    }
    for label, expected_output in expected_outputs.items():
        tolerance = 2.0e-7 if label == "WENO5 FP32" else 5.0e-15
        if not np.allclose(outputs[label], expected_output, rtol=0.0, atol=tolerance):
            raise AssertionError(f"{label} fixed-stencil inference signature mismatch")

    limits = (("WENO5 FP64", defect5, 1.0e-14), ("WENO5 FP32", defect5f32, 2.0e-7), ("WENO7 FP64", defect7, 1.0e-14))
    messages = []
    for label, defect, limit in limits:
        if not np.isfinite(defect) or defect > limit:
            raise AssertionError(f"{label} reflection defect {defect:.3e} > {limit:.3e}")
        messages.append(f"reflection ok: {label}, max defect={defect:.3e}")
    return messages


def verify_copy_inventory() -> list[str]:
    path = REPOSITORY / "scripts" / "copy_inventory.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    for record in records:
        destination = REPOSITORY / record["destination"]
        if not destination.is_file():
            raise AssertionError(f"missing copied artifact: {destination}")
        if destination.stat().st_size != record["bytes"]:
            raise AssertionError(f"copied artifact size mismatch: {destination}")
        if sha256(destination) != record["sha256"]:
            raise AssertionError(f"copied artifact hash mismatch: {destination}")
    return [f"copy inventory ok: {len(records)} immutable files"]


def verify_imports() -> list[str]:
    checks = (
        (
            SNAPSHOT,
            "import teacherfree_lab_weno5_v20_distance_balanced.train_weno5_v20",
            "WENO5 FP64 training",
        ),
        (
            SNAPSHOT / "teacherfree_lab_weno5_v20_distance_balanced_mlp_f32_fast",
            "import train",
            "WENO5 FP32 training",
        ),
        (
            SNAPSHOT / "teacherfree_lab_weno7_rk4_distance_balanced_fast-2",
            "import train",
            "WENO7 FP64 training",
        ),
        (
            SNAPSHOT,
            "import for_paper_results.run_gste; import for_paper_results.run_sod; "
            "import for_paper_results.run_vortex; import for_paper_results.run_quadrant; "
            "import for_paper_results.run_double_mach; import runpy; "
            "runpy.run_path('shockbubble_ma3_t0001_cfl04_server/for_paper_results/"
            "run_weno5_shockbubble.py')",
            "paper Euler/advection runners",
        ),
        (
            SNAPSHOT,
            "import weno_z_borges_p2_results.run_gste; "
            "import weno_z_borges_p2_results.run_riemann_1d; "
            "import weno_z_borges_p2_results.run_shockbubble_2d; "
            "import weno_z_borges_p2_results.run_shockbubble_3d",
            "WENO-Z baselines",
        ),
        (
            SNAPSHOT,
            "import warp_weno5_3d_rk3.run_shockbubble_ma3_mlp; "
            "import warp_weno7_3d_rk4.run_shockbubble_ma3",
            "three-dimensional runners",
        ),
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(SNAPSHOT) + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    messages: list[str] = []
    for cwd, code, label in checks:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"import smoke test failed for {label}:\n{result.stderr.strip()}"
            )
        messages.append(f"import ok: {label}")
    return messages


def verify_experiments() -> list[str]:
    root = REPOSITORY / "experiments"
    actual = sorted(path.name for path in root.iterdir() if path.is_dir())
    if actual != list(EXPERIMENTS):
        raise AssertionError(f"experiment directories differ: {actual}")
    required = ("README.md", "config.yaml", "run.sh", "plot.sh", "provenance.json")
    artifact_count = 0
    for name in EXPERIMENTS:
        experiment = root / name
        for filename in required:
            path = experiment / filename
            if not path.is_file():
                raise AssertionError(f"missing experiment file: {path}")
        for script in (experiment / "run.sh", experiment / "plot.sh"):
            if not os.access(script, os.X_OK):
                raise AssertionError(f"experiment wrapper is not executable: {script}")
            syntax = subprocess.run(
                ["bash", "-n", str(script)], capture_output=True, text=True, check=False
            )
            if syntax.returncode != 0:
                raise AssertionError(
                    f"invalid shell wrapper {script}: {syntax.stderr.strip()}"
                )
        readme = (experiment / "README.md").read_text(encoding="utf-8")
        for heading in (
            "## Purpose",
            "## Configuration",
            "## Initial condition",
            "## Methods",
            "## Results",
            "## Plot construction",
            "## Reproduction",
            "## Provenance and data policy",
        ):
            if heading not in readme:
                raise AssertionError(f"{experiment}/README.md lacks {heading}")
        config = (experiment / "config.yaml").read_text(encoding="utf-8")
        for key in (
            "experiment_id:",
            "selected_checkpoints:",
            "weno5_sr_fp64:",
            "weno5_sr_fp32:",
            "weno7_sr_fp64:",
            "reflection_symmetrised_inference: true",
            "raw_data_in_release: false",
        ):
            if key not in config:
                raise AssertionError(f"{experiment}/config.yaml lacks {key}")
        provenance = json.loads((experiment / "provenance.json").read_text(encoding="utf-8"))
        if provenance.get("experiment") != name:
            raise AssertionError(f"provenance experiment mismatch: {name}")
        for key in (
            "selected_checkpoints",
            "run_entry_point",
            "plot_entry_points",
            "manuscript_source",
            "manuscript_location",
            "artifacts",
        ):
            if key not in provenance:
                raise AssertionError(f"{experiment}/provenance.json lacks {key}")
        if set(provenance["selected_checkpoints"]) != {
            "weno5_sr_fp64", "weno5_sr_fp32", "weno7_sr_fp64"
        }:
            raise AssertionError(f"selected checkpoint provenance mismatch: {name}")
        for artifact in provenance.get("artifacts", []):
            path = REPOSITORY / artifact["destination"]
            if not path.is_file() or sha256(path) != artifact["sha256"]:
                raise AssertionError(f"experiment artifact mismatch: {path}")
            artifact_count += 1
        if not (experiment / "figures").is_dir() or not (experiment / "tables").is_dir():
            raise AssertionError(f"missing figures/tables directory: {name}")
        figures = list((experiment / "figures").glob("*"))
        tables = list((experiment / "tables").glob("*"))
        if name != "14_mixed_precision_timing" and not figures:
            raise AssertionError(f"no publication figure archived for {name}")
        if name == "14_mixed_precision_timing" and not (
            experiment / "figures" / "README.md"
        ).is_file():
            raise AssertionError("timing experiment must document its table-only figure policy")
        if not tables:
            raise AssertionError(f"no compact result table archived for {name}")
    return [f"experiment layout ok: {len(EXPERIMENTS)} experiments, {artifact_count} artifacts"]


def verify_release_policy() -> list[str]:
    allowed_run_files = {Path(value) for value in MODEL_ALIASES.values()}
    files = release_files()
    for path in files:
        relative = path.relative_to(REPOSITORY)
        if path.stat().st_size > MAX_BYTES:
            raise AssertionError(f"file exceeds 50 MiB: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise AssertionError(f"forbidden numerical file type: {relative}")
        if any(part in FORBIDDEN_PARTS for part in relative.parts):
            raise AssertionError(f"forbidden generated-data directory: {relative}")
        if any(part in {"runs", "checkpoints"} for part in relative.parts) and relative not in allowed_run_files:
            raise AssertionError(f"unselected run/checkpoint artifact: {relative}")
        if path.is_symlink():
            raise AssertionError(f"release contains a symlink rather than a copy: {relative}")
    npz = {path.relative_to(REPOSITORY) for path in files if path.suffix == ".npz"}
    expected_npz = {Path("models") / name for name in MODELS} | allowed_run_files
    if npz != expected_npz:
        raise AssertionError(f"unexpected NPZ set: {sorted(map(str, npz ^ expected_npz))}")
    third_party = REPOSITORY / "experiments/10_double_mach_reflection/figures/castro_wenoz_orders_reference.png"
    if third_party.exists():
        raise AssertionError("third-party WENO-Z screenshot must not be redistributed")
    return [f"release policy ok: {len(files)} files, largest <= 50 MiB, no raw fields"]


def verify_pdf() -> list[str]:
    path = REPOSITORY / "paper/WENO_SR_current_draft.pdf"
    if sha256(path) != "08e67fc3e34d68b5ee120105a6e4ec7c6d0cbcf9ae065698d1cebee2f07e344d":
        raise AssertionError("manuscript PDF hash mismatch")
    data = path.read_bytes()
    if not data.startswith(b"%PDF-") or b"%%EOF" not in data[-4096:]:
        raise AssertionError("manuscript does not have a valid PDF envelope")
    pdfinfo = subprocess.run(
        ["pdfinfo", str(path)], capture_output=True, text=True, check=False
    ) if shutil_which("pdfinfo") else None
    if pdfinfo is not None and pdfinfo.returncode != 0:
        raise AssertionError(f"pdfinfo rejected the manuscript: {pdfinfo.stderr.strip()}")
    return [f"PDF ok: {path.name}, {path.stat().st_size} bytes"]


def shutil_which(command: str) -> str | None:
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(directory) / command
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def write_manifest() -> Path:
    path = REPOSITORY / "MANIFEST.sha256"
    lines = [
        f"{sha256(file)}  {file.relative_to(REPOSITORY).as_posix()}"
        for file in release_files()
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def verify_manifest() -> list[str]:
    path = REPOSITORY / "MANIFEST.sha256"
    if not path.is_file():
        raise AssertionError("MANIFEST.sha256 is missing; run with --write-manifest")
    listed: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        listed[relative] = digest
    files = {file.relative_to(REPOSITORY).as_posix(): file for file in release_files()}
    if set(listed) != set(files):
        raise AssertionError("manifest file set does not match the release")
    for relative, file in files.items():
        if listed[relative] != sha256(file):
            raise AssertionError(f"manifest hash mismatch: {relative}")
    return [f"manifest ok: {len(files)} files"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--skip-reflection", action="store_true")
    args = parser.parse_args()

    checks = [
        verify_models,
        verify_copy_inventory,
        verify_imports,
        verify_experiments,
        verify_release_policy,
        verify_pdf,
    ]
    if not args.skip_reflection:
        checks.insert(1, verify_reflection)
    for check in checks:
        for message in check():
            print(message)
    if args.write_manifest:
        print(f"wrote {write_manifest()}")
    for message in verify_manifest():
        print(message)
    print("release verification passed")


if __name__ == "__main__":
    main()
