from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp


PARAMETER_NAMES = ("w1", "b1", "w2", "b2", "w3", "b3", "w4", "b4")
EXPECTED_SHAPES = {
    "w1": (1, 5, 10),
    "b1": (1, 10),
    "w2": (1, 10, 6),
    "b2": (1, 6),
    "w3": (1, 6, 6),
    "b3": (1, 6),
    "w4": (1, 6, 3),
    "b4": (1, 3),
}
ARCHITECTURE_TAG = "shared_direct_beta_ratio_5_10_6_6_3"
REFLECTION_FORMULA = "0.5*(M(x)+P*M(P*x))"


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
            "scale_feature": "clip((log10(delta_max/q_scale)+4)/4,0,1)",
            "raw_step": self.metadata.get("raw_step"),
            "recipe": self.metadata.get("recipe"),
        }


def read_checkpoint(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in PARAMETER_NAMES if name not in archive.files]
        if missing:
            raise ValueError(f"{path} is missing MLP arrays: {missing}")
        wrong = {
            name: archive[name].shape
            for name, expected in EXPECTED_SHAPES.items()
            if archive[name].shape != expected
        }
        if wrong:
            raise ValueError(f"{path} has incompatible WENO5 MLP shapes: {wrong}")
        payload = {
            name: np.ascontiguousarray(archive[name], dtype=np.float64)
            for name in PARAMETER_NAMES
        }
        metadata: dict[str, object] = {}
        if "meta_json" in archive.files:
            metadata = json.loads(str(archive["meta_json"]))

    architecture = str(metadata.get("mlp_architecture", ""))
    if ARCHITECTURE_TAG not in architecture:
        raise ValueError(
            f"{path} does not declare the required {ARCHITECTURE_TAG} architecture"
        )
    if metadata.get("deployment_requires_reflection_symmetrization") is not True:
        raise ValueError(f"{path} does not require reflection-symmetric deployment")
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

