from __future__ import annotations

import argparse
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from bsc_gexit_surface_sampled import (  # noqa: E402
    binary_entropy,
    binary_entropy_axis,
    write_outputs,
)


P_ATOL = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge independent coupled BSC GEXIT repeat results."
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--repeat", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to the base JSON directory.",
    )
    return parser.parse_args()


def load_result(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def result_samples(result: dict) -> int:
    samples = result.get("samples")
    if not isinstance(samples, int) or samples <= 0:
        raise ValueError("BSC GEXIT repeat result must contain a positive integer samples field")
    return samples


def point_array(result: dict, key: str) -> np.ndarray:
    return np.array([float(point[key]) for point in result["points"]], dtype=float)


def point_array_default(result: dict, key: str, default: float = 0.0) -> np.ndarray:
    return np.array(
        [float(point.get(key, default)) for point in result["points"]],
        dtype=float,
    )


def assert_compatible(base: dict, repeat: dict) -> None:
    for field in ("n", "k", "distance"):
        if base["code"].get(field) != repeat["code"].get(field):
            raise ValueError(f"code field {field!r} differs between base and repeat")

    base_p = point_array(base, "p")
    repeat_p = point_array(repeat, "p")
    if base_p.shape != repeat_p.shape or not np.allclose(base_p, repeat_p, atol=P_ATOL, rtol=0.0):
        raise ValueError("base and repeat must use the same p grid")

    if not base.get("paired_derivative") or not repeat.get("paired_derivative"):
        raise ValueError("only coupled/paired BSC GEXIT results can be merged")

    required = (
        "posterior_x_error",
        "posterior_x_error_stderr",
        "posterior_x_class",
        "posterior_x_class_stderr",
        "posterior_x_class_component_norm_dp",
        "posterior_x_class_component_norm_dp_stderr",
    )
    for result_name, result in (("base", base), ("repeat", repeat)):
        missing = [key for key in required if key not in result["points"][0]]
        if missing:
            raise ValueError(f"{result_name} result is missing columns: {missing}")


def combine_mean_stderr(
    mean_a: np.ndarray,
    stderr_a: np.ndarray,
    n_a: int,
    mean_b: np.ndarray,
    stderr_b: np.ndarray,
    n_b: int,
) -> tuple[np.ndarray, np.ndarray]:
    total = n_a + n_b
    mean = (n_a * mean_a + n_b * mean_b) / total
    if total <= 1:
        return mean, np.zeros_like(mean)

    var_a = stderr_a * stderr_a * n_a if n_a > 1 else np.zeros_like(mean_a)
    var_b = stderr_b * stderr_b * n_b if n_b > 1 else np.zeros_like(mean_b)
    m2 = (
        max(n_a - 1, 0) * var_a
        + max(n_b - 1, 0) * var_b
        + n_a * (mean_a - mean) ** 2
        + n_b * (mean_b - mean) ** 2
    )
    variance = np.maximum(m2 / (total - 1), 0.0)
    return mean, np.sqrt(variance / total)


def source_summary(result: dict, path: Path | None) -> dict:
    summary = {
        "samples": result.get("samples"),
        "seed": result.get("seed"),
        "method": result.get("method"),
    }
    if path is not None:
        summary["path"] = str(path)
    if "merged_sources" in result:
        summary["merged_sources"] = result["merged_sources"]
    return summary


def merge_result_dicts(
    base: dict,
    repeat: dict,
    *,
    base_path: Path | None = None,
    repeat_path: Path | None = None,
) -> dict:
    assert_compatible(base, repeat)
    n_a = result_samples(base)
    n_b = result_samples(repeat)
    total = n_a + n_b
    n = int(base["code"]["n"])
    p = point_array(base, "p")

    h_s_a = n * np.array([binary_entropy(float(value)) for value in p]) - point_array(
        base,
        "posterior_x_error",
    )
    h_s_b = n * np.array([binary_entropy(float(value)) for value in p]) - point_array(
        repeat,
        "posterior_x_error",
    )
    h_s, h_s_stderr = combine_mean_stderr(
        h_s_a,
        point_array(base, "posterior_x_error_stderr"),
        n_a,
        h_s_b,
        point_array(repeat, "posterior_x_error_stderr"),
        n_b,
    )
    class_entropy, class_stderr = combine_mean_stderr(
        point_array(base, "posterior_x_class"),
        point_array(base, "posterior_x_class_stderr"),
        n_a,
        point_array(repeat, "posterior_x_class"),
        point_array(repeat, "posterior_x_class_stderr"),
        n_b,
    )
    dydp, dydp_stderr = combine_mean_stderr(
        point_array(base, "posterior_x_class_component_norm_dp"),
        point_array_default(base, "posterior_x_class_component_norm_dp_stderr"),
        n_a,
        point_array(repeat, "posterior_x_class_component_norm_dp"),
        point_array_default(repeat, "posterior_x_class_component_norm_dp_stderr"),
        n_b,
    )
    dydp = np.maximum(dydp, 0.0)
    peak_idx = int(np.argmax(dydp))
    peak_value = float(dydp[peak_idx])
    scale = 1.0 / peak_value if peak_value > 0.0 else 1.0

    rows = []
    for idx, p_value in enumerate(p):
        raw_error = n * binary_entropy(float(p_value)) - h_s[idx]
        saved = raw_error - class_entropy[idx]
        rows.append(
            {
                "p": float(p_value),
                "posterior_x_error": float(raw_error),
                "posterior_x_class": float(class_entropy[idx]),
                "posterior_x_saved_by_stabilizers": float(saved),
                "posterior_x_error_stderr": float(h_s_stderr[idx]),
                "posterior_x_class_stderr": float(class_stderr[idx]),
                "posterior_x_class_component_norm": float(class_entropy[idx] / n),
                "posterior_x_class_component_norm_stderr": float(class_stderr[idx] / n),
                "posterior_x_class_component_norm_dp": float(dydp[idx]),
                "posterior_x_class_component_norm_dp_stderr": float(dydp_stderr[idx]),
                "scaled_posterior_x_class_component_norm_dp": float(dydp[idx] * scale),
                "scaled_posterior_x_class_component_norm_dp_stderr": float(
                    dydp_stderr[idx] * scale
                ),
            }
        )

    merged = deepcopy(base)
    merged["samples"] = total
    merged["seed"] = None
    merged["workers"] = None
    merged["batch_count"] = None
    merged["grid"] = {
        "p": [float(value) for value in p],
        "t": [float(value) for value in binary_entropy_axis(p)],
    }
    merged["scaling"] = {
        "peak_p": float(rows[peak_idx]["p"]),
        "peak_gexit": peak_value,
        "scale_to_unit_peak": scale,
        "trapezoid_area_component_norm": float(np.trapezoid(dydp, p)),
        "target_area_k_over_n": base["code"]["k"] / base["code"]["n"],
    }
    merged["points"] = rows
    merged["merged_sources"] = [
        source_summary(base, base_path),
        source_summary(repeat, repeat_path),
    ]
    merged["job"] = {
        "kind": "merged entropy-centered coupled BSC GEXIT surface run",
        "distance": base["code"].get("distance"),
        "samples": total,
        "paired_derivative": True,
    }
    return merged


def main() -> None:
    args = parse_args()
    base = load_result(args.base)
    merged = base
    base_path = args.base
    for repeat_path in args.repeat:
        repeat = load_result(repeat_path)
        merged = merge_result_dicts(
            merged,
            repeat,
            base_path=base_path,
            repeat_path=repeat_path,
        )
        base_path = None
    out_dir = args.out_dir if args.out_dir is not None else args.base.parent
    write_outputs(merged, out_dir, out_dir / "tikz")


if __name__ == "__main__":
    main()
