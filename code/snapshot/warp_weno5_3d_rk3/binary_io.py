from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np


HEADER = struct.Struct("=IIId")
COMPONENT_NAMES = ("rho", "u", "v", "w", "p")


def read_step(path: str | Path) -> tuple[float, np.ndarray]:
    path = Path(path)
    with path.open("rb") as stream:
        raw = stream.read(HEADER.size)
        if len(raw) != HEADER.size:
            raise ValueError(f"truncated step header: {path}")
        nz, ny, nx, time = HEADER.unpack(raw)
        count = int(nz) * int(ny) * int(nx) * 5
        values = np.fromfile(stream, dtype=np.float64, count=count)
        trailing = stream.read(1)
    if values.size != count:
        raise ValueError(f"truncated payload: expected {count}, got {values.size}")
    if trailing:
        raise ValueError(f"unexpected trailing bytes in {path}")
    return float(time), values.reshape((nz, ny, nx, 5))


def write_step(path: str | Path, time: float, primitive: np.ndarray) -> None:
    path = Path(path)
    primitive = np.ascontiguousarray(primitive, dtype=np.float64)
    if primitive.ndim != 4 or primitive.shape[-1] != 5:
        raise ValueError("primitive must have shape (nz, ny, nx, 5)")
    nz, ny, nx, _ = primitive.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(HEADER.pack(nz, ny, nx, float(time)))
        primitive.tofile(stream)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def files_bitwise_equal(left: str | Path, right: str | Path) -> bool:
    left = Path(left)
    right = Path(right)
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as lhs, right.open("rb") as rhs:
        while True:
            a = lhs.read(1024 * 1024)
            b = rhs.read(1024 * 1024)
            if a != b:
                return False
            if not a:
                return True


def _normalized_l1(error: np.ndarray, reference: np.ndarray) -> float:
    numerator = float(np.sum(error))
    denominator = float(np.sum(np.abs(reference)))
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def _ordered_int64(values: np.ndarray) -> np.ndarray:
    bits = np.asarray(values, dtype=np.float64).view(np.uint64)
    sign = np.uint64(1) << np.uint64(63)
    return np.where((bits & sign) != 0, ~bits, bits | sign)


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    oa = _ordered_int64(a)
    ob = _ordered_int64(b)
    hi = np.maximum(oa, ob)
    lo = np.minimum(oa, ob)
    return hi - lo


def compare_steps(candidate: str | Path, reference: str | Path) -> dict[str, object]:
    candidate = Path(candidate)
    reference = Path(reference)
    tc, uc = read_step(candidate)
    tr, ur = read_step(reference)
    if uc.shape != ur.shape:
        raise ValueError(f"shape mismatch: {uc.shape} != {ur.shape}")

    same = uc.view(np.uint64) == ur.view(np.uint64)
    abs_error = np.abs(uc - ur)
    ulp = ulp_distance(uc, ur)
    unequal = ~same
    finite_pair = np.isfinite(uc) & np.isfinite(ur)

    report: dict[str, object] = {
        "candidate": str(candidate),
        "reference": str(reference),
        "candidate_sha256": sha256(candidate),
        "reference_sha256": sha256(reference),
        "file_bitwise_identical": files_bitwise_equal(candidate, reference),
        "array_bitwise_identical": bool(np.all(same)),
        "header": {
            "shape_equal": uc.shape == ur.shape,
            "candidate_time": tc,
            "reference_time": tr,
            "time_bitwise_equal": struct.pack("=d", tc) == struct.pack("=d", tr),
        },
        "unequal_count": int(np.count_nonzero(unequal)),
        "finite_pair_count": int(np.count_nonzero(finite_pair)),
        "normalized_l1": _normalized_l1(abs_error, ur),
        "mean_absolute_error": float(np.mean(abs_error)),
        "max_absolute_error": float(np.max(abs_error)),
        "max_ulp": int(np.max(ulp)),
        "components": {},
    }

    components: dict[str, object] = {}
    for c, name in enumerate(COMPONENT_NAMES):
        mask = unequal[..., c]
        err = abs_error[..., c]
        comp_ulp = ulp[..., c]
        components[name] = {
            "unequal_count": int(np.count_nonzero(mask)),
            "mean_absolute_error": float(np.mean(err)),
            "max_absolute_error": float(np.max(err)),
            "normalized_l1": _normalized_l1(err, ur[..., c]),
            "max_ulp": int(np.max(comp_ulp)),
        }
    report["components"] = components

    if np.any(unequal):
        flat_first = int(np.flatnonzero(unequal.ravel())[0])
        first = tuple(int(v) for v in np.unravel_index(flat_first, uc.shape))
        flat_max = int(np.argmax(abs_error.ravel()))
        maximum = tuple(int(v) for v in np.unravel_index(flat_max, uc.shape))

        def detail(index: tuple[int, ...]) -> dict[str, object]:
            z, y, x, c = index
            return {
                "index": {"z": z, "y": y, "x": x, "component": COMPONENT_NAMES[c]},
                "candidate": float(uc[index]),
                "reference": float(ur[index]),
                "absolute_error": float(abs_error[index]),
                "ulp": int(ulp[index]),
            }

        report["first_difference"] = detail(first)
        report["largest_absolute_difference"] = detail(maximum)
    return report


def write_comparison(report: dict[str, object], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii")
