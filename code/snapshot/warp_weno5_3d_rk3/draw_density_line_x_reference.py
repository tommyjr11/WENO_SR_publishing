#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import multiprocessing as mp
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed


# ────────────────────────────────────────────────────────────────────────────────
# 全局期刊绘图样式配置 (对齐你给的那种 JCP 风格)
# ────────────────────────────────────────────────────────────────────────────────
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10.0,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 4.2,
        "ytick.major.size": 4.2,
        "xtick.minor.size": 2.4,
        "ytick.minor.size": 2.4,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "legend.frameon": False,
        "axes.unicode_minus": False,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    }
)


# ────────────────────────────────────────────────────────────────────────────────
# 配置区域
# ────────────────────────────────────────────────────────────────────────────────
DATA_DIR_DEFAULT = Path("/home/ruijie/data_shock_bubble_3D_4th/data_3D")
OUTPUT_DIR = Path(
    "warp_weno5_3d_rk3/runs/ma3_t0001_cfl025_N224x88x88/"
    "reference_line_profiles"
)

PHYS_X_MIN, PHYS_X_MAX = 0.0, 0.225
PHYS_Y_MIN, PHYS_Y_MAX = 0.0, 0.089
PHYS_Z_MIN, PHYS_Z_MAX = 0.0, 0.089

# 剖线方向:
# "x" -> 固定 y, z，沿 x 画
# "y" -> 固定 x, z，沿 y 画
# "z" -> 固定 x, y，沿 z 画
PROFILE_DIRECTION = "y"

# 目标坐标的取值方式:
# "physical"   -> 直接按物理坐标解释，例如 0.065
# "normalized" -> 按 [0, 1] 归一化坐标解释，例如 0.65
# "auto"       -> 优先当物理坐标，不在物理域内时再按归一化坐标解释
# 对你这个问题最稳的是 auto:
# - 写 0.065 会被当成 physical
# - 写 0.65  会被当成 normalized
YZ_COORD_MODE = "auto"
X_TARGET = 0.115
Y_TARGET = 0.0445
Z_TARGET = 0.0445

# 输入文件列表: ("标签名", "数据目录或.bin文件", step)
# 例如:
# ("Present", "data_3D", 7048)
# ("Method-A", "data_shock", 905)
# ("Method-B", "data_shock/step_0905.bin", 905)
# 后面你加别的方法，直接继续往这里加就行。
FILES_TO_COMPARE = [
    (
        "Reference",
        "/home/ruijie/data_shock_bubble_3D_4th/data_3D",
        7048,
    ),
    (
        "WENO5_JS_224",
        "warp_weno5_3d_rk3/runs/ma3_t0001_cfl025_N224x88x88/"
        "weno5_js/step_0823.bin",
        823,
    ),
    (
        "WENO5_JS_RK3_448",
        "classical_3d_N448_a100/results/"
        "weno5_js_rk3_N448x176x176_t0001_cfl025/step_1743.bin",
        1743,
    ),
    (
        "WENO5_SR_FP64_224",
        "warp_weno5_3d_rk3/runs/ma3_t0001_cfl025_N224x88x88/"
        "weno5_sr_f64_v20_step012250/step_0863.bin",
        863,
    ),
    (
        "WENO5_SR_FP32_224",
        "warp_weno5_3d_rk3/runs/ma3_t0001_cfl025_N224x88x88/"
        "weno5_sr_f32_v20_step016500/step_0863.bin",
        863,
    ),
    (
        "WENO7_JS_RK4_224",
        "warp_weno7_3d_rk4/runs/ma3_js_rk4_t0001_cfl025_N224x88x88/"
        "step_0841.bin",
        841,
    ),
    (
        "WENO7_JS_RK4_448",
        "classical_3d_N448_a100/results/"
        "weno7_js_rk4_N448x176x176_t0001_cfl025/step_1764.bin",
        1764,
    ),
    (
        "WENO7_SR_FP64_RK4_224",
        "warp_weno7_3d_rk4/runs/ma3_sr_f64_step016750_rk4_t0001_cfl025_N224x88x88/"
        "step_0871.bin",
        871,
    ),
]

# 可选样式覆盖；不写就自动分配不同颜色
STYLE_OVERRIDES = {
    "Reference": {"color": "black", "linestyle": "-", "linewidth": 2.05, "zorder": 12},
    "WENO5_JS_224": {
        "color": "#777777", "linestyle": (0, (4.5, 2.2)),
        "linewidth": 1.25, "zorder": 7,
    },
    "WENO5_JS_RK3_448": {
        "color": "#009E73", "linestyle": (0, (1.0, 1.8)), "linewidth": 1.30,
        "marker": "D", "markersize": 2.5, "markevery": 28,
        "markerfacecolor": "white", "markeredgewidth": 0.60, "zorder": 8,
    },
    "WENO5_SR_FP64_224": {
        "color": "#0072B2", "linestyle": "-", "linewidth": 1.55,
        "marker": "s", "markersize": 2.8, "markevery": 12,
        "markerfacecolor": "white", "markeredgewidth": 0.65, "zorder": 10,
    },
    "WENO5_SR_FP32_224": {
        "color": "#D55E00", "linestyle": "-.", "linewidth": 1.45,
        "marker": "^", "markersize": 2.9, "markevery": 12,
        "markerfacecolor": "white", "markeredgewidth": 0.65, "zorder": 10,
    },
    "WENO7_JS_RK4_224": {
        "color": "#882255", "linestyle": (0, (4.5, 1.8)),
        "linewidth": 1.35, "zorder": 9,
    },
    "WENO7_JS_RK4_448": {
        "color": "#CC79A7", "linestyle": (0, (6.0, 2.2)), "linewidth": 1.30,
        "marker": "X", "markersize": 2.6, "markevery": 28,
        "markerfacecolor": "white", "markeredgewidth": 0.60, "zorder": 8,
    },
    "WENO7_SR_FP64_RK4_224": {
        "color": "#332288", "linestyle": "-", "linewidth": 1.65,
        "marker": "P", "markersize": 2.9, "markevery": 12,
        "markerfacecolor": "white", "markeredgewidth": 0.65, "zorder": 11,
    },
}
DISPLAY_LABEL_OVERRIDES = {
    "Reference": "TR1W ref., 1120 × 440²",
    "WENO5_JS_224": "WENO5-JS/RK3, 224 × 88²",
    "WENO5_JS_RK3_448": "WENO5-JS/RK3, 448 × 176²",
    "WENO5_SR_FP64_224": "WENO5-SR (FP64)/RK3, 224 × 88²",
    "WENO5_SR_FP32_224": "WENO5-SR (FP32)/RK3, 224 × 88²",
    "WENO7_JS_RK4_224": "WENO7-JS/RK4, 224 × 88²",
    "WENO7_JS_RK4_448": "WENO7-JS/RK4, 448 × 176²",
    "WENO7_SR_FP64_RK4_224": "WENO7-SR (FP64)/RK4, 224 × 88²",
}

VAR_NAME = "rho"
VAR_LABEL_MAP = {
    "rho": r"Density $\rho$",
    "u": r"Velocity $u$",
    "p": r"Pressure $p$",
}
DISPLAY_VAR_NAME = VAR_LABEL_MAP.get(VAR_NAME, VAR_NAME)

FIGSIZE = (8.4, 5.45)
DPI = 600
LEGEND_LOC = "center left"
TITLE = None
ENABLE_GRID = True

LINE_LIMITS_FULL = None
VALUE_LIMITS_FULL = None

# 按你给的脚本风格，预留局部放大图
# 格式: ("文件后缀", (xmin, xmax), (ymin, ymax))
ZOOM_WINDOWS = [
    # ("zoom1", (0.08, 0.12), (4.85, 4.95)),
    # ("zoom2", (0.14, 0.18), (4.90, 5.05)),
]

OUTPUT_BASENAME = "density_line_reference_compare"
SAVE_PNG = True
SAVE_PDF = True
SAVE_ERROR_CSV = True

# 为了和 224 基准网格公平比较：
# - 448 (2x) 这类偶数倍细化，固定截线位置取左右两个细网格中心平均
# - 1120 (5x) 这类奇数倍细化，粗网格中心正好落在中间细网格中心上
FAIR_SLICE_AVERAGING = True
BASE_COMPARE_NX = 224
BASE_COMPARE_NY = 88
BASE_COMPARE_NZ = 88

# 颜色循环: 首条默认黑色，后面用色盲友好的颜色
COLOR_CYCLE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#7F3C1D",
]
LINESTYLE_CYCLE = ["-"]
DEFAULT_LINEWIDTH = 1.15
REFERENCE_LINEWIDTH = 1.35

# 3D 数据很大，跨方法并行不要开太大，默认 1 最稳
COMPARE_WORKERS = 1
MAX_IO_WORKERS = 40
READ_CACHE = {}


# ────────────────────────────────────────────────────────────────────────────────
# 核心逻辑 (读取方式沿用 draw_time.py 的 MPI 拼接思路)
# ────────────────────────────────────────────────────────────────────────────────
def _read_one_mpi_file(path: Path):
    try:
        rank = int(path.stem.split("_")[-1])
    except ValueError:
        rank = 0

    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=np.uint32, count=3)
        if header.size != 3:
            raise ValueError(f"{path}: 无法读取头部 nz, ny, nx")
        nz, ny, nx = header.astype(np.int64)

        t_arr = np.fromfile(f, dtype=np.float64, count=1)
        if t_arr.size != 1:
            raise ValueError(f"{path}: 无法读取时间 t")
        t = float(t_arr[0])

        raw_len = nz * ny * nx * 5
        raw = np.fromfile(f, dtype=np.float64)
        if raw.size != raw_len:
            expected_bytes = 3 * 4 + 8 + raw_len * 8
            actual_bytes = path.stat().st_size
            raise ValueError(
                f"{path}: header 给出的尺寸是 (nz={nz}, ny={ny}, nx={nx})，"
                f"按 5 个 float64 变量应有 {raw_len} 个 double "
                f"(文件大小 {expected_bytes} bytes)，"
                f"但实际只有 {raw.size} 个 double "
                f"(文件大小 {actual_bytes} bytes)。"
                f"这通常说明这个文件不是当前这套 5 分量 3D 二进制格式，"
                f"或者你现在指向了错误的目录/文件。"
            )

    rho_local = raw.reshape(nz, ny, nx, 5)[..., 0].astype(np.float32)
    return {"rank": rank, "rho": rho_local, "nx": nx, "t": t}


def resolve_input_files(source: Path, step: int):
    if source.suffix == ".bin":
        if not source.exists():
            raise FileNotFoundError(f"找不到指定文件: {source}")
        return "single", [source]

    if not source.exists():
        raise FileNotFoundError(f"找不到目录: {source}")

    single_file = source / f"step_{step:04d}.bin"
    if single_file.exists():
        return "single", [single_file]

    mpi_pattern = re.compile(rf"step_{step:04d}_(\d+)\.bin$")
    mpi_files = []
    for path in source.glob(f"step_{step:04d}_*.bin"):
        match = mpi_pattern.fullmatch(path.name)
        if match:
            mpi_files.append((int(match.group(1)), path))

    if mpi_files:
        mpi_files.sort(key=lambda item: item[0])
        return "mpi", [path for _, path in mpi_files]

    similar = sorted(path.name for path in source.glob(f"step_{step:04d}*.bin"))
    if similar:
        raise FileNotFoundError(
            f"在 {source} 里没有找到标准命名的 step_{step:04d}.bin 或 step_{step:04d}_rank.bin，"
            f"但找到了这些相近文件: {similar}"
        )

    raise FileNotFoundError(f"在 {source} 里找不到 step {step:04d} 的数据文件")


def read_step_rho(step: int, source: Path):
    cache_key = (str(source.resolve()), int(step))
    if cache_key in READ_CACHE:
        rho, t = READ_CACHE[cache_key]
        print(f"[I/O] 使用只读缓存: {source} | step={step:04d}")
        return rho, t

    io_mode, files = resolve_input_files(source, step)

    if io_mode == "single":
        single_file = files[0]
        print(f"[I/O] 读取单文件: {single_file}")
        res = _read_one_mpi_file(single_file)
        print(f"[I/O] 完成: shape={res['rho'].shape}, t={res['t']:.8e}")
        READ_CACHE[cache_key] = (res["rho"], res["t"])
        return READ_CACHE[cache_key]

    print(f"[I/O] 读取 step {step:04d}: {len(files)} 个 MPI 分块 | {source}")
    results = []
    workers = min(len(files), MAX_IO_WORKERS)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_read_one_mpi_file, path) for path in files]
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: item["rank"])

    nz = results[0]["rho"].shape[0]
    ny = results[0]["rho"].shape[1]
    nx_total = sum(item["nx"] for item in results)
    t0 = results[0]["t"]

    rho_global = np.empty((nz, ny, nx_total), dtype=np.float32)
    curr = 0
    for item in results:
        nx_local = item["nx"]
        rho_global[:, :, curr : curr + nx_local] = item["rho"]
        curr += nx_local

    print(f"[I/O] 拼接完成: shape={rho_global.shape}, t={t0:.8e}")
    READ_CACHE[cache_key] = (rho_global, t0)
    return READ_CACHE[cache_key]


def build_cell_centers(coord_min: float, coord_max: float, ncell: int) -> np.ndarray:
    dx = (coord_max - coord_min) / ncell
    return coord_min + (np.arange(ncell, dtype=np.float64) + 0.5) * dx


def resolve_target_location(
    value: float,
    mode: str,
    coord_min: float,
    coord_max: float,
    ncell: int,
    axis_name: str,
):
    centers = build_cell_centers(coord_min, coord_max, ncell)

    if mode == "physical":
        target_physical = float(value)
        used_mode = "physical"
    elif mode == "normalized":
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{axis_name}={value} 不在归一化范围 [0, 1] 内")
        target_physical = coord_min + value * (coord_max - coord_min)
        used_mode = "normalized"
    else:
        if coord_min <= value <= coord_max:
            target_physical = float(value)
            used_mode = "physical"
        elif 0.0 <= value <= 1.0:
            target_physical = coord_min + value * (coord_max - coord_min)
            used_mode = "normalized"
        else:
            raise ValueError(
                f"{axis_name}={value} 既不在物理范围 [{coord_min}, {coord_max}]，"
                f"也不在归一化范围 [0, 1] 内"
            )

    idx = int(np.argmin(np.abs(centers - target_physical)))
    return idx, target_physical, float(centers[idx]), used_mode


def resolve_axis_meta(axis_name: str, ncell: int):
    if axis_name == "x":
        coord_min, coord_max, target = PHYS_X_MIN, PHYS_X_MAX, X_TARGET
        base_ncell = BASE_COMPARE_NX
    elif axis_name == "y":
        coord_min, coord_max, target = PHYS_Y_MIN, PHYS_Y_MAX, Y_TARGET
        base_ncell = BASE_COMPARE_NY
    elif axis_name == "z":
        coord_min, coord_max, target = PHYS_Z_MIN, PHYS_Z_MAX, Z_TARGET
        base_ncell = BASE_COMPARE_NZ
    else:
        raise ValueError(f"不支持的轴方向: {axis_name}")

    if FAIR_SLICE_AVERAGING and ncell % base_ncell == 0:
        ref_idx, target_physical, ref_physical, used_mode = resolve_target_location(
            target,
            YZ_COORD_MODE,
            coord_min,
            coord_max,
            base_ncell,
            axis_name,
        )
        ratio = ncell // base_ncell
        centers = build_cell_centers(coord_min, coord_max, ncell)

        if ratio % 2 == 0:
            left = ref_idx * ratio + ratio // 2 - 1
            right = left + 1
            sample_indices = [left, right]
            selection = "pair_average"
        else:
            center = ref_idx * ratio + ratio // 2
            sample_indices = [center]
            selection = "single_center"

        actual_physical = float(np.mean(centers[sample_indices]))
        return {
            "axis": axis_name,
            "requested": target,
            "index": ref_idx,
            "target_physical": target_physical,
            "actual_physical": actual_physical,
            "mode": used_mode,
            "sample_indices": sample_indices,
            "selection": selection,
            "ratio": ratio,
            "reference_physical": ref_physical,
        }

    idx, target_physical, actual_physical, used_mode = resolve_target_location(
        target,
        YZ_COORD_MODE,
        coord_min,
        coord_max,
        ncell,
        axis_name,
    )
    return {
        "axis": axis_name,
        "requested": target,
        "index": idx,
        "target_physical": target_physical,
        "actual_physical": actual_physical,
        "mode": used_mode,
        "sample_indices": [idx],
        "selection": "nearest",
        "ratio": 1,
        "reference_physical": actual_physical,
    }


def get_profile_settings(direction: str):
    if direction == "x":
        return {
            "line_axis": "x",
            "fixed_axes": ("y", "z"),
            "xlabel": "X (m)",
            "default_limits": (PHYS_X_MIN, PHYS_X_MAX),
        }
    if direction == "y":
        return {
            "line_axis": "y",
            "fixed_axes": ("x", "z"),
            "xlabel": "Y (m)",
            "default_limits": (PHYS_Y_MIN, PHYS_Y_MAX),
        }
    if direction == "z":
        return {
            "line_axis": "z",
            "fixed_axes": ("x", "y"),
            "xlabel": "Z (m)",
            "default_limits": (PHYS_Z_MIN, PHYS_Z_MAX),
        }
    raise ValueError(f"PROFILE_DIRECTION 只支持 x / y / z，当前为 {direction}")


def extract_profile(rho: np.ndarray):
    nz, ny, nx = rho.shape

    axis_meta = {
        "x": resolve_axis_meta("x", nx),
        "y": resolve_axis_meta("y", ny),
        "z": resolve_axis_meta("z", nz),
    }

    settings = get_profile_settings(PROFILE_DIRECTION)

    if PROFILE_DIRECTION == "x":
        line_axis = build_cell_centers(PHYS_X_MIN, PHYS_X_MAX, nx)
        z_indices = axis_meta["z"]["sample_indices"]
        y_indices = axis_meta["y"]["sample_indices"]
        slab = rho[np.ix_(z_indices, y_indices, np.arange(nx))]
        profile = slab.mean(axis=(0, 1)).astype(np.float64)
    elif PROFILE_DIRECTION == "y":
        line_axis = build_cell_centers(PHYS_Y_MIN, PHYS_Y_MAX, ny)
        z_indices = axis_meta["z"]["sample_indices"]
        x_indices = axis_meta["x"]["sample_indices"]
        slab = rho[np.ix_(z_indices, np.arange(ny), x_indices)]
        profile = slab.mean(axis=(0, 2)).astype(np.float64)
    else:
        line_axis = build_cell_centers(PHYS_Z_MIN, PHYS_Z_MAX, nz)
        y_indices = axis_meta["y"]["sample_indices"]
        x_indices = axis_meta["x"]["sample_indices"]
        slab = rho[np.ix_(np.arange(nz), y_indices, x_indices)]
        profile = slab.mean(axis=(1, 2)).astype(np.float64)

    meta = {
        "profile_direction": PROFILE_DIRECTION,
        "line_axis": settings["line_axis"],
        "fixed_axes": settings["fixed_axes"],
        "xlabel": settings["xlabel"],
        "default_limits": settings["default_limits"],
        "axes": axis_meta,
    }
    return line_axis, profile, meta


def _read_worker(task):
    label, source_str, step = task
    source = Path(source_str)

    try:
        rho, t = read_step_rho(step, source)
        line_axis, profile, meta = extract_profile(rho)
        return {
            "label": label,
            "step": step,
            "time": t,
            "line_axis": line_axis,
            "profile": profile,
            "meta": meta,
        }
    except Exception as exc:
        print(f"读取失败 {label} | step={step} | {source}: {exc}")
        return None


def get_line_style(label: str, index: int):
    if label in STYLE_OVERRIDES:
        style = STYLE_OVERRIDES[label].copy()
        style.setdefault("linewidth", DEFAULT_LINEWIDTH)
        style.setdefault("linestyle", "-")
        style.setdefault("solid_capstyle", "round")
        style.setdefault("solid_joinstyle", "round")
        return style

    label_lower = label.lower()
    if "present" in label_lower or "reference" in label_lower:
        return {
            "color": "black",
            "linestyle": "-",
            "linewidth": REFERENCE_LINEWIDTH,
            "zorder": 10,
            "solid_capstyle": "round",
            "solid_joinstyle": "round",
        }

    color_index = max(index - 1, 0)
    return {
        "color": COLOR_CYCLE[color_index % len(COLOR_CYCLE)],
        "linestyle": LINESTYLE_CYCLE[index % len(LINESTYLE_CYCLE)],
        "linewidth": DEFAULT_LINEWIDTH,
        "zorder": 8,
        "solid_capstyle": "round",
        "solid_joinstyle": "round",
    }


def ordinal_suffix(value: int) -> str:
    if 10 <= value % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")


def format_case_label(label: str) -> str:
    if label in DISPLAY_LABEL_OVERRIDES:
        return DISPLAY_LABEL_OVERRIDES[label]

    match = re.fullmatch(r"([A-Za-z0-9]+)_(\d+)(?:st|nd|rd|th)_(\d+)", label)
    if not match:
        return label.replace("_", " ")

    scheme, order_text, resolution_text = match.groups()
    order = int(order_text)
    resolution = int(resolution_text)
    return f"{scheme} ({order}{ordinal_suffix(order)}-order, X = {resolution})"


def is_reference_label(label: str) -> bool:
    label_lower = label.lower()
    return "present" in label_lower or "reference" in label_lower


def compute_profile_total_error(reference_result: dict, result: dict) -> dict:
    ref_axis = reference_result["line_axis"]
    ref_profile = reference_result["profile"]
    axis = result["line_axis"]
    profile = result["profile"]

    common_min = max(float(ref_axis[0]), float(axis[0]))
    common_max = min(float(ref_axis[-1]), float(axis[-1]))
    common_axis = ref_axis[(ref_axis >= common_min) & (ref_axis <= common_max)]
    if common_axis.size < 2:
        raise ValueError("参考曲线与目标曲线的公共区间不足，无法计算总误差")

    ref_common = np.interp(common_axis, ref_axis, ref_profile)
    profile_common = np.interp(common_axis, axis, profile)
    diff = profile_common - ref_common

    total_abs_error = float(np.trapezoid(np.abs(diff), common_axis))
    ref_norm = float(np.trapezoid(np.abs(ref_common), common_axis))
    relative_error = total_abs_error / ref_norm if ref_norm > 0.0 else np.nan

    return {
        "total_abs_error": total_abs_error,
        "relative_error": relative_error,
        "max_abs_error": float(np.max(np.abs(diff))),
        "common_min": common_min,
        "common_max": common_max,
    }


def compute_profile_error_curve(reference_result: dict, result: dict):
    ref_axis = reference_result["line_axis"]
    ref_profile = reference_result["profile"]
    axis = result["line_axis"]
    profile = result["profile"]

    mask = (ref_axis >= axis[0]) & (ref_axis <= axis[-1])
    common_axis = ref_axis[mask]
    ref_common = ref_profile[mask]
    profile_common = np.interp(common_axis, axis, profile)
    return common_axis, np.abs(profile_common - ref_common)


def save_error_report(rows: list[dict], stem: str):
    if not SAVE_ERROR_CSV or not rows:
        return None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"{stem}_errors.csv"
    fieldnames = [
        "label",
        "reference",
        "step",
        "total_abs_error",
        "relative_error",
        "max_abs_error",
        "common_min",
        "common_max",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def save_current_figure(fig, stem: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved = []

    if SAVE_PNG:
        png_path = OUTPUT_DIR / f"{stem}.png"
        fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
        saved.append(png_path)

    if SAVE_PDF:
        pdf_path = OUTPUT_DIR / f"{stem}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        saved.append(pdf_path)

    return saved


def fmt_value(value: float) -> str:
    return f"{value:.4f}".replace("-", "m").replace(".", "p")


def get_requested_value(axis_name: str) -> float:
    if axis_name == "x":
        return X_TARGET
    if axis_name == "y":
        return Y_TARGET
    if axis_name == "z":
        return Z_TARGET
    raise ValueError(f"不支持的轴方向: {axis_name}")


def main():
    read_tasks = [(label, str(Path(source)), step) for label, source, step in FILES_TO_COMPARE]

    if not read_tasks:
        print("没有配置任何要绘制的数据，程序退出。")
        return

    settings = get_profile_settings(PROFILE_DIRECTION)
    fixed_axes = settings["fixed_axes"]
    print(
        f"开始绘制 {PROFILE_DIRECTION} 方向 {DISPLAY_VAR_NAME} 剖线: "
        f"{fixed_axes[0].upper()}_TARGET={get_requested_value(fixed_axes[0])}, "
        f"{fixed_axes[1].upper()}_TARGET={get_requested_value(fixed_axes[1])}, "
        f"mode={YZ_COORD_MODE}"
    )

    if COMPARE_WORKERS > 1 and len(read_tasks) > 1:
        workers = min(COMPARE_WORKERS, len(read_tasks))
        print(f"启动跨方法并行读取: {workers} 个进程")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_read_worker, read_tasks))
    else:
        results = [_read_worker(task) for task in read_tasks]

    results = [item for item in results if item is not None]
    if not results:
        print("没有成功读取到任何数据，程序退出。")
        return

    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    grid_spec = fig.add_gridspec(2, 1, height_ratios=(3.55, 1.0), hspace=0.055)
    ax = fig.add_subplot(grid_spec[0])
    ax_error = fig.add_subplot(grid_spec[1], sharex=ax)

    actual_meta = results[0]["meta"]
    reference_result = next((item for item in results if is_reference_label(item["label"])), results[0])
    reference_display_label = format_case_label(reference_result["label"])
    error_rows = []
    for axis in fixed_axes:
        axis_meta = actual_meta["axes"][axis]
        sample_desc = ",".join(str(idx) for idx in axis_meta["sample_indices"])
        print(
            f"[Slice] {axis}={axis_meta['requested']} -> {axis_meta['mode']} -> "
            f"target={axis_meta['target_physical']:.8f} m -> "
            f"sampled={axis_meta['actual_physical']:.8f} m "
            f"(samples={sample_desc}, mode={axis_meta['selection']})"
        )

    for idx, result in enumerate(results):
        label = result["label"]
        display_label = format_case_label(label)
        line_axis = result["line_axis"]
        profile = result["profile"]
        style = get_line_style(label, idx)
        result_fixed = result["meta"]["fixed_axes"]

        print(f"\n处理: {label}")
        fixed_desc = ", ".join(
            f"{axis}={result['meta']['axes'][axis]['actual_physical']:.8f} m "
            f"(samples={','.join(str(idx) for idx in result['meta']['axes'][axis]['sample_indices'])})"
            for axis in result_fixed
        )
        print(f"  -> step={result['step']}, {fixed_desc}")

        if result is reference_result:
            print(f"  -> reference profile: {display_label}")
        else:
            error_metrics = compute_profile_total_error(reference_result, result)
            result["error_metrics"] = error_metrics
            error_rows.append(
                {
                    "label": display_label,
                    "reference": reference_display_label,
                    "step": result["step"],
                    "total_abs_error": f"{error_metrics['total_abs_error']:.8e}",
                    "relative_error": f"{error_metrics['relative_error']:.8e}",
                    "max_abs_error": f"{error_metrics['max_abs_error']:.8e}",
                    "common_min": f"{error_metrics['common_min']:.8e}",
                    "common_max": f"{error_metrics['common_max']:.8e}",
                }
            )
            print(
                "  -> total error vs reference: "
                f"L1={error_metrics['total_abs_error']:.6e}, "
                f"relative={error_metrics['relative_error']:.3%}, "
                f"Linf={error_metrics['max_abs_error']:.6e}"
            )

        ax.plot(line_axis, profile, label=display_label, **style)

    error_panel_labels = {
        "WENO5_JS_224",
        "WENO5_JS_RK3_448",
        "WENO5_SR_FP64_224",
        "WENO5_SR_FP32_224",
        "WENO7_JS_RK4_224",
        "WENO7_JS_RK4_448",
        "WENO7_SR_FP64_RK4_224",
    }
    for idx, result in enumerate(results):
        label = result["label"]
        if label not in error_panel_labels:
            continue
        error_axis, error_profile = compute_profile_error_curve(reference_result, result)
        error_style = get_line_style(label, idx)
        error_style["linewidth"] = max(1.0, error_style.get("linewidth", 1.0) - 0.18)
        for marker_key in (
            "marker", "markersize", "markevery", "markerfacecolor", "markeredgewidth"
        ):
            error_style.pop(marker_key, None)
        error_style["alpha"] = 0.92
        ax_error.plot(error_axis, error_profile, **error_style)

    ax.set_axisbelow(True)
    ax.set_ylabel(DISPLAY_VAR_NAME, labelpad=5, fontsize=11.0)
    ax.tick_params(which="major", pad=4)
    ax.tick_params(labelbottom=False)

    ax_error.set_xlabel(actual_meta["xlabel"].replace("Y", "y"), labelpad=4, fontsize=11.0)
    ax_error.set_ylabel(r"$|\rho-\rho_{\mathrm{ref}}|$", labelpad=4, fontsize=10.0)
    ax_error.tick_params(which="major", pad=3)
    ax_error.set_ylim(bottom=0.0)

    if TITLE:
        fig.suptitle(TITLE, fontsize=12.5, y=0.988)

    if ENABLE_GRID:
        for plot_axis in (ax, ax_error):
            plot_axis.grid(
                which="major",
                linestyle=(0, (2.0, 2.0)),
                linewidth=0.38,
                color="#C8CED8",
                alpha=0.42,
            )

    handles, labels = ax.get_legend_handles_labels()
    handle_by_label = dict(zip(labels, handles))
    legend_key_order = [
        "Reference",
        "WENO7_JS_RK4_224",
        "WENO5_JS_224",
        "WENO7_SR_FP64_RK4_224",
        "WENO5_SR_FP64_224",
        "WENO5_JS_RK3_448",
        "WENO5_SR_FP32_224",
        "WENO7_JS_RK4_448",
    ]
    labels = [format_case_label(key) for key in legend_key_order]
    handles = [handle_by_label[label] for label in labels]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        fontsize=7.15,
        ncol=4,
        handlelength=2.45,
        columnspacing=1.05,
        handletextpad=0.42,
        borderaxespad=0.0,
        bbox_to_anchor=(0.5, 0.940),
    )

    line_limits = LINE_LIMITS_FULL if LINE_LIMITS_FULL is not None else actual_meta["default_limits"]
    if line_limits is not None:
        ax.set_xlim(*line_limits)
    if VALUE_LIMITS_FULL is not None:
        ax.set_ylim(*VALUE_LIMITS_FULL)

    ax.text(0.012, 0.965, "(a)", transform=ax.transAxes, ha="left", va="top", fontsize=10.5)
    ax_error.text(
        0.012, 0.90, "(b)", transform=ax_error.transAxes, ha="left", va="top", fontsize=10.5
    )
    fig.subplots_adjust(left=0.10, right=0.988, bottom=0.10, top=0.825, hspace=0.055)

    fixed_suffix = "_".join(
        f"{axis}{fmt_value(get_requested_value(axis))}" for axis in fixed_axes
    )
    base_stem = f"{OUTPUT_BASENAME}_line{PROFILE_DIRECTION}_{fixed_suffix}_{YZ_COORD_MODE}"

    saved_paths = save_current_figure(fig, base_stem + "_full")
    print("\n[1/1] 全景图已保存:")
    for path in saved_paths:
        print(f"  - {path}")

    error_report_path = save_error_report(error_rows, base_stem + "_full")
    if error_report_path is not None:
        print("[Error] 误差报告已保存:")
        print(f"  - {error_report_path}")

    original_xlim = ax.get_xlim()
    original_ylim = ax.get_ylim()

    for idx, (suffix, xlim, ylim) in enumerate(ZOOM_WINDOWS, start=1):
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        zoom_paths = save_current_figure(fig, base_stem + f"_{suffix}")
        print(f"[zoom {idx}] 局部放大图已保存:")
        for path in zoom_paths:
            print(f"  - {path}")

    ax.set_xlim(*original_xlim)
    ax.set_ylim(*original_ylim)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Reference density line cuts for the three formal 3-D Warp runs; "
            "the original data and plotting script remain read-only"
        )
    )
    parser.add_argument("--x", type=float, nargs="+", default=[0.085, 0.097, 0.114])
    parser.add_argument("--z", type=float, default=0.0445)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    mp.freeze_support()
    args = parse_args()
    OUTPUT_DIR = args.output_dir
    Z_TARGET = args.z
    for requested_x in args.x:
        X_TARGET = requested_x
        TITLE = (
            rf"Ma = 3 shock-bubble interaction: "
            rf"$x = {requested_x:.3f}$ m, $z = {Z_TARGET:.4f}$ m"
        )
        main()
