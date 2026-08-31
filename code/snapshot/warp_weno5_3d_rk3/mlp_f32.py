from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

from .mlp import ARCHITECTURE_TAG, EXPECTED_SHAPES, PARAMETER_NAMES, REFLECTION_FORMULA


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_parameter_dtype(path: str | Path) -> np.dtype:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with np.load(resolved, allow_pickle=False) as archive:
        missing = [name for name in PARAMETER_NAMES if name not in archive.files]
        if missing:
            raise ValueError(f"{resolved} is missing MLP arrays: {missing}")
        dtypes = {np.dtype(archive[name].dtype) for name in PARAMETER_NAMES}
    if len(dtypes) != 1:
        raise ValueError(f"{resolved} mixes MLP parameter dtypes: {sorted(map(str, dtypes))}")
    dtype = dtypes.pop()
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"{resolved} uses unsupported MLP parameter dtype {dtype}")
    return dtype


@dataclass(frozen=True)
class MlpFloat32Parameters:
    path: Path
    sha256: str
    metadata: dict[str, object]
    w1: object
    b1: object
    w2: object
    b2: object
    w3: object
    b3: object
    w4: object
    b4: object

    def kernel_inputs(self) -> list[object]:
        return [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3, self.w4, self.b4]

    def manifest(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "parameter_dtype": "float32",
            "solver_state_dtype": "float64",
            "feature_preprocessing_dtype": "float64",
            "reflection_average_dtype": "float64",
            "weno_normalization_dtype": "float64",
            "architecture": ARCHITECTURE_TAG,
            "reflection_formula": REFLECTION_FORMULA,
            "scale_feature": "clip((log10(delta_max/q_scale)+4)/4,0,1)",
            "raw_step": self.metadata.get("raw_step"),
            "recipe": self.metadata.get("recipe"),
        }


def read_float32_checkpoint(
    path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with np.load(resolved, allow_pickle=False) as archive:
        missing = [name for name in PARAMETER_NAMES if name not in archive.files]
        if missing:
            raise ValueError(f"{resolved} is missing MLP arrays: {missing}")
        wrong = {
            name: archive[name].shape
            for name, expected in EXPECTED_SHAPES.items()
            if archive[name].shape != expected
        }
        if wrong:
            raise ValueError(f"{resolved} has incompatible WENO5 MLP shapes: {wrong}")
        wrong_dtype = {
            name: str(archive[name].dtype)
            for name in PARAMETER_NAMES
            if archive[name].dtype != np.float32
        }
        if wrong_dtype:
            raise ValueError(f"{resolved} is not a native float32 MLP checkpoint: {wrong_dtype}")
        payload = {
            name: np.ascontiguousarray(archive[name], dtype=np.float32)
            for name in PARAMETER_NAMES
        }
        metadata: dict[str, object] = {}
        if "meta_json" in archive.files:
            metadata = json.loads(str(archive["meta_json"]))

    architecture = str(metadata.get("mlp_architecture", ""))
    if ARCHITECTURE_TAG not in architecture:
        raise ValueError(
            f"{resolved} does not declare the required {ARCHITECTURE_TAG} architecture"
        )
    required = {
        "deployment_requires_reflection_symmetrization": True,
        "mlp_parameter_dtype": "float32",
        "mlp_activation_dtype": "float32",
        "feature_preprocessing_dtype": "float64",
        "reflection_average_dtype": "float64",
        "weno_normalization_dtype": "float64",
        "solver_state_dtype": "float64",
    }
    mismatch = {
        key: metadata.get(key)
        for key, expected in required.items()
        if metadata.get(key) != expected
    }
    if mismatch:
        raise ValueError(f"{resolved} has incompatible mixed-precision metadata: {mismatch}")
    return payload, metadata


def load_mlp_float32_parameters(
    path: str | Path,
    device: str | wp.context.Device,
) -> MlpFloat32Parameters:
    resolved = Path(path).resolve()
    payload, metadata = read_float32_checkpoint(resolved)
    arrays = {
        name: wp.array(payload[name], dtype=wp.float32, device=device, requires_grad=False)
        for name in PARAMETER_NAMES
    }
    return MlpFloat32Parameters(
        path=resolved,
        sha256=_sha256(resolved),
        metadata=metadata,
        **arrays,
    )
