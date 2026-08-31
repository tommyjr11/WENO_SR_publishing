from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "for_paper_results"
RAW = PACKAGE / "raw"
FIGURES = PACKAGE / "figures"
TABLES = PACKAGE / "tables"
LOGS = PACKAGE / "logs"

GSTE_WENO5_V20_T10_CFL06 = ROOT / (
    "teacherfree_lab_weno5_v20_distance_balanced/runs/"
    "apost_weno5_v20_distance_balanced_cfl05_200k/"
    "gste_validation_step_012250/step_012250/result_cfl_0p6.npz"
)

DOUBLE_MACH_WENO5_JS_STATE = (
    ROOT / "plots/WENO5_MLP/weno_double_reflective_1200/weno5_classical.npy"
)
DOUBLE_MACH_WENO5_V20_RESULT = ROOT / (
    "plots/WENO5_MLP/teacherfree_weno5_v20_refsym/double_mach/"
    "apost_weno5_v20_distance_balanced_model_step_012250_"
    "double_mach_1200x300_t02_hllc_cfl04/mlp_double_mach_results.npz"
)


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    family: str
    time_integrator: str
    model: Path | None
    mlp_precision: str | None
    color: str
    linestyle: str


METHODS = {
    "weno5_js": Method(
        "weno5_js", "WENO5-JS-RK3", "weno5", "SSPRK3", None, None,
        "#4D4D4D", "--",
    ),
    "weno5_sr_f64": Method(
        "weno5_sr_f64", "WENO5-SR-RK3", "weno5", "SSPRK3",
        ROOT / (
            "teacherfree_lab_weno5_v20_distance_balanced/runs/"
            "apost_weno5_v20_distance_balanced_cfl05_200k/checkpoints/"
            "model_step_012250.npz"
        ),
        "float64", "#0072B2", "-",
    ),
    "weno5_sr_f32": Method(
        "weno5_sr_f32", "WENO5-SR-FP32-RK3", "weno5_mixed", "SSPRK3",
        ROOT / (
            "teacherfree_lab_weno5_v20_distance_balanced_mlp_f32_fast/runs/"
            "apost_weno5_v20_mlp_f32_fast_200k/checkpoints/"
            "model_step_016500.npz"
        ),
        "float32", "#009E73", "-.",
    ),
    "weno7_js": Method(
        "weno7_js", "WENO7-JS-RK4", "weno7", "SSP-RK4", None, None,
        "#D55E00", "--",
    ),
    "weno7_sr_f64": Method(
        "weno7_sr_f64", "WENO7-SR-RK4", "weno7", "SSP-RK4",
        ROOT / (
            "teacherfree_lab_weno7_rk4_distance_balanced_fast/runs/"
            "apost_weno7_rk4_distance_balanced_fast_4090_200k/checkpoints/"
            "model_step_016750.npz"
        ),
        "float64", "#CC79A7", "-",
    ),
}

EULER_METHODS = tuple(METHODS)


def ensure_output_dirs() -> None:
    for path in (RAW, FIGURES, TABLES, LOGS):
        path.mkdir(parents=True, exist_ok=True)


def validate_models() -> None:
    missing = [str(m.model) for m in METHODS.values() if m.model is not None and not m.model.is_file()]
    if missing:
        raise FileNotFoundError("missing selected checkpoint(s):\n" + "\n".join(missing))
