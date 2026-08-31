from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp


PARAMETER_NAMES = ("w1", "b1", "w2", "b2", "w3", "b3", "w4", "b4")
EXPECTED_SHAPES = {
    "w1": (1, 6, 24),
    "b1": (1, 24),
    "w2": (1, 24, 16),
    "b2": (1, 16),
    "w3": (1, 16, 16),
    "b3": (1, 16),
    "w4": (1, 16, 4),
    "b4": (1, 4),
}
ARCHITECTURE_TAG = "reflection_sym_direct_beta_ratio_6_24_16_16_4"
REFLECTION_FORMULA = "0.5*(M(x)+P4*M(P6*x))"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class MlpParameters:
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
            "parameter_dtype": "float64",
            "architecture": ARCHITECTURE_TAG,
            "reflection_formula": REFLECTION_FORMULA,
            "scale_feature": "clip((log10(delta_max/q_scale)+16)/16,0,1)",
            "raw_step": self.metadata.get("raw_step"),
            "recipe": self.metadata.get("recipe"),
        }


def read_checkpoint(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
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
            raise ValueError(f"{resolved} has incompatible WENO7 MLP shapes: {wrong}")
        payload = {
            name: np.ascontiguousarray(archive[name], dtype=np.float64)
            for name in PARAMETER_NAMES
        }
        metadata: dict[str, object] = {}
        if "meta_json" in archive.files:
            metadata = json.loads(str(archive["meta_json"]))

    architecture = str(metadata.get("mlp_architecture", ""))
    if architecture != ARCHITECTURE_TAG:
        raise ValueError(
            f"{resolved} declares {architecture!r}; expected {ARCHITECTURE_TAG!r}"
        )
    if metadata.get("deployment_requires_reflection_symmetrization") is not True:
        raise ValueError(f"{resolved} does not require reflection-symmetric deployment")
    precision = str(metadata.get("precision", "float64"))
    if precision != "float64":
        raise ValueError(f"{resolved} is {precision}; this solver requires the FP64 WENO7 MLP")
    return payload, metadata


def load_mlp_parameters(path: str | Path, device: str | wp.context.Device) -> MlpParameters:
    resolved = Path(path).resolve()
    payload, metadata = read_checkpoint(resolved)
    arrays = {
        name: wp.array(payload[name], dtype=wp.float64, device=device, requires_grad=False)
        for name in PARAMETER_NAMES
    }
    return MlpParameters(
        path=resolved,
        sha256=_sha256(resolved),
        metadata=metadata,
        **arrays,
    )
