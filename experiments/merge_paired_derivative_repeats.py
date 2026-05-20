from __future__ import annotations

import argparse
import json
import math
import sys
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exit_curve_experiments import (  # noqa: E402
    derivative_areas,
    plot_paired_derivative_result,
    write_paired_derivative_outputs,
)


POINT_KEYS = {"p", "p_low", "p_high", "runs"}
P_ATOL = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge independent paired-derivative repeat results into a base result."
    )
    parser.add_argument("--base", type=Path, required=True, help="Base paired-derivative JSON.")
    parser.add_argument("--repeat", type=Path, required=True, help="Repeat paired-derivative JSON.")
    parser.add_argument(
        "--merge-ps",
        type=float,
        nargs="+",
        required=True,
        help="p values to merge from the repeat into the base.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to the base JSON directory.",
    )
    parser.add_argument("--plot", action="store_true", help="Regenerate the paired plot too.")
    return parser.parse_args()


def load_result(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def p_key(value: float) -> float:
    return round(float(value), 12)


def points_by_p(result: dict) -> dict[float, dict]:
    return {p_key(point["p"]): point for point in result["points"]}


def assert_same_window(base_point: dict, repeat_point: dict) -> None:
    for key in ("p", "p_low", "p_high"):
        if not math.isclose(
            float(base_point[key]),
            float(repeat_point[key]),
            rel_tol=0.0,
            abs_tol=P_ATOL,
        ):
            raise ValueError(
                f"cannot merge p={base_point['p']}: {key} differs "
                f"({base_point[key]} vs {repeat_point[key]})"
            )


def combine_mean_stderr(
    mean_a: float,
    stderr_a: float,
    n_a: int,
    mean_b: float,
    stderr_b: float,
    n_b: int,
) -> tuple[float, float]:
    total = n_a + n_b
    mean = (n_a * mean_a + n_b * mean_b) / total

    if total <= 1:
        return mean, 0.0

    var_a = (stderr_a * math.sqrt(n_a)) ** 2 if n_a > 1 else 0.0
    var_b = (stderr_b * math.sqrt(n_b)) ** 2 if n_b > 1 else 0.0
    m2 = (
        max(n_a - 1, 0) * var_a
        + max(n_b - 1, 0) * var_b
        + n_a * (mean_a - mean) ** 2
        + n_b * (mean_b - mean) ** 2
    )
    variance = m2 / (total - 1)
    return mean, math.sqrt(max(variance, 0.0) / total)


def combine_point(base_point: dict, repeat_point: dict) -> dict:
    assert_same_window(base_point, repeat_point)
    n_a = int(base_point["runs"])
    n_b = int(repeat_point["runs"])
    merged = deepcopy(base_point)

    for key, value_a in base_point.items():
        if key in POINT_KEYS or key.endswith("_stderr"):
            continue
        if key not in repeat_point:
            continue

        stderr_key = f"{key}_stderr"
        value_b = repeat_point[key]
        if stderr_key in base_point and stderr_key in repeat_point:
            mean, stderr = combine_mean_stderr(
                float(value_a),
                float(base_point[stderr_key]),
                n_a,
                float(value_b),
                float(repeat_point[stderr_key]),
                n_b,
            )
            merged[key] = mean
            merged[stderr_key] = stderr
        else:
            merged[key] = (n_a * float(value_a) + n_b * float(value_b)) / (n_a + n_b)

    merged["runs"] = n_a + n_b
    return merged


def merge_results(
    base: dict,
    repeat: dict,
    merge_ps: list[float],
    *,
    base_path: Path,
    repeat_path: Path,
) -> dict:
    base_by_p = points_by_p(base)
    repeat_by_p = points_by_p(repeat)
    selected = {p_key(value) for value in merge_ps}

    missing = sorted(selected - set(base_by_p))
    if missing:
        raise ValueError(f"base result is missing p values: {missing}")
    missing = sorted(selected - set(repeat_by_p))
    if missing:
        raise ValueError(f"repeat result is missing p values: {missing}")

    merged = deepcopy(base)
    merged_points = []
    for point in base["points"]:
        key = p_key(point["p"])
        if key in selected:
            merged_points.append(combine_point(point, repeat_by_p[key]))
        else:
            merged_points.append(deepcopy(point))

    merged["points"] = merged_points
    merged["areas"] = derivative_areas(merged_points)
    merged["elapsed_seconds"] = float(base.get("elapsed_seconds", 0.0)) + float(
        repeat.get("elapsed_seconds", 0.0)
    )
    merged["config"] = deepcopy(base["config"])
    merged["config"]["runs"] = "mixed"
    merged["config"]["merged_repeat"] = {
        "base": base_path.name,
        "repeat": repeat_path.name,
        "merged_ps": sorted(selected),
        "base_runs": base["config"].get("runs"),
        "repeat_runs": repeat["config"].get("runs"),
        "repeat_seed": repeat["config"].get("seed"),
    }
    return merged


def main() -> None:
    args = parse_args()
    base = load_result(args.base)
    repeat = load_result(args.repeat)
    merged = merge_results(
        base,
        repeat,
        args.merge_ps,
        base_path=args.base,
        repeat_path=args.repeat,
    )

    out_dir = args.out_dir if args.out_dir is not None else args.base.parent
    json_path, csv_path = write_paired_derivative_outputs(merged, out_dir)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    if args.plot:
        png_path = plot_paired_derivative_result(merged, out_dir)
        print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
