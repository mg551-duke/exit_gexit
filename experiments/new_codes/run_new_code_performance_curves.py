from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from ldpc import BpOsdDecoder
except ImportError as exc:  # pragma: no cover - dependency availability is environment-specific
    BpOsdDecoder = None
    LDPC_IMPORT_ERROR = exc
else:
    LDPC_IMPORT_ERROR = None

from experiments.exit_curve_experiments import (
    ExactClassCache,
    exact_x_class_target,
    exact_z_class_target,
    load_code,
    load_exact_class_cache,
    mask_from_bool,
)


DEFAULT_CODES_DIR = REPO_ROOT / "codes" / "new"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "experiments" / "new_codes" / "performance_curves"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate logical performance curves for the new qLDPC codes. "
            "BEC curves use exact erasure class ambiguity. BSC curves use BP+OSD."
        )
    )
    parser.add_argument("--codes-dir", type=Path, default=DEFAULT_CODES_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--channel", choices=["bec", "bsc", "both"], default="both")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--bec-runs", type=int, default=20000)
    parser.add_argument("--bec-points", type=int, default=41)
    parser.add_argument("--bsc-runs", type=int, default=10000)
    parser.add_argument("--bsc-points", type=int, default=21)
    parser.add_argument("--bsc-p-max", type=float, default=0.2)
    parser.add_argument("--bp-max-iter", type=int, default=0)
    parser.add_argument("--bp-method", choices=["minimum_sum", "product_sum"], default="minimum_sum")
    parser.add_argument("--ms-scaling-factor", type=float, default=0.625)
    parser.add_argument("--osd-method", choices=["OSD_0", "OSD_E", "OSD_CS"], default="OSD_CS")
    parser.add_argument("--osd-order", type=int, default=2)
    parser.add_argument("--confidence-z", type=float, default=1.959963984540054)
    return parser.parse_args()


def safe_stem(path: Path) -> str:
    stem = path.stem.replace("[[", "").replace("]]", "")
    return re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")


def write_points(result: dict, json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    fields = sorted({key for point in result["points"] for key in point})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["points"])


def wilson_interval(failures: int, runs: int, z: float) -> tuple[float, float]:
    if runs <= 0:
        return 0.0, 0.0
    phat = failures / runs
    denom = 1.0 + z * z / runs
    centre = (phat + z * z / (2.0 * runs)) / denom
    radius = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * runs)) / runs) / denom
    return max(0.0, centre - radius), min(1.0, centre + radius)


def mean_interval(values: list[float], z: float) -> tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    stderr = float(arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0
    return mean, stderr, max(0.0, mean - z * stderr), min(1.0, mean + z * stderr)


def code_metadata(path: Path) -> dict[str, object]:
    code = load_code(path)
    return {
        "path": str(path),
        "name": path.stem,
        "n": code.n,
        "k": code.k,
        "rank_hx": code.rank_hx,
        "rank_hz": code.rank_hz,
    }


def run_bec_performance(
    code_path: Path,
    p_values: np.ndarray,
    runs: int,
    seed: int,
    confidence_z: float,
) -> dict:
    cache = load_exact_class_cache(code_path)
    rng = np.random.default_rng(seed)

    x_ml_values: list[list[float]] = [[] for _ in p_values]
    z_ml_values: list[list[float]] = [[] for _ in p_values]
    total_ml_values: list[list[float]] = [[] for _ in p_values]
    x_uncorrectable = np.zeros(len(p_values), dtype=np.int64)
    z_uncorrectable = np.zeros(len(p_values), dtype=np.int64)
    total_uncorrectable = np.zeros(len(p_values), dtype=np.int64)

    for _ in range(runs):
        thresholds = rng.random(cache.code.n)
        for p_idx, p in enumerate(p_values):
            erased_mask = mask_from_bool(thresholds < p)
            x_dim = int(exact_x_class_target(cache, erased_mask, use_logicals=True))
            z_dim = int(exact_z_class_target(cache, erased_mask, use_logicals=True))
            total_dim = x_dim + z_dim

            x_ml = 1.0 - 2.0 ** (-x_dim)
            z_ml = 1.0 - 2.0 ** (-z_dim)
            total_ml = 1.0 - 2.0 ** (-total_dim)

            x_ml_values[p_idx].append(x_ml)
            z_ml_values[p_idx].append(z_ml)
            total_ml_values[p_idx].append(total_ml)
            x_uncorrectable[p_idx] += int(x_dim > 0)
            z_uncorrectable[p_idx] += int(z_dim > 0)
            total_uncorrectable[p_idx] += int(total_dim > 0)

    points = []
    for p_idx, p in enumerate(p_values):
        row: dict[str, float | int] = {"p": float(p), "runs": int(runs)}
        for prefix, values in (
            ("x_ml_logical_failure", x_ml_values[p_idx]),
            ("z_ml_logical_failure", z_ml_values[p_idx]),
            ("total_ml_logical_failure", total_ml_values[p_idx]),
        ):
            mean, stderr, low, high = mean_interval(values, confidence_z)
            row[prefix] = mean
            row[f"{prefix}_stderr"] = stderr
            row[f"{prefix}_ci_low"] = low
            row[f"{prefix}_ci_high"] = high

        for prefix, failures in (
            ("x_uncorrectable_rate", int(x_uncorrectable[p_idx])),
            ("z_uncorrectable_rate", int(z_uncorrectable[p_idx])),
            ("total_uncorrectable_rate", int(total_uncorrectable[p_idx])),
        ):
            low, high = wilson_interval(failures, runs, confidence_z)
            row[prefix] = failures / runs
            row[f"{prefix}_failures"] = failures
            row[f"{prefix}_ci_low"] = low
            row[f"{prefix}_ci_high"] = high
        points.append(row)

    return {
        "code": code_metadata(code_path),
        "config": {
            "channel": "BEC",
            "runs": int(runs),
            "seed": int(seed),
            "p_values": [float(p) for p in p_values],
            "confidence_z": float(confidence_z),
            "estimator": "exact erasure class ambiguity",
            "x_ml_logical_failure": "E[1 - 2^{-H(C_X|E,S)}]",
            "z_ml_logical_failure": "E[1 - 2^{-H(C_Z|E,S)}]",
            "total_ml_logical_failure": "E[1 - 2^{-(H(C_X|E,S)+H(C_Z|E,S))}]",
        },
        "points": points,
    }


def build_bposd_decoder(checks: np.ndarray, p: float, args: argparse.Namespace) -> BpOsdDecoder:
    if BpOsdDecoder is None:
        raise RuntimeError(f"ldpc package is required for BSC curves: {LDPC_IMPORT_ERROR}")
    max_iter = int(args.bp_max_iter) if args.bp_max_iter > 0 else int(checks.shape[1])
    return BpOsdDecoder(
        checks.astype(np.uint8),
        error_rate=float(p),
        max_iter=max_iter,
        bp_method=args.bp_method,
        ms_scaling_factor=float(args.ms_scaling_factor),
        schedule="parallel",
        omp_thread_count=1,
        osd_method=args.osd_method,
        osd_order=int(args.osd_order),
        input_vector_type="syndrome",
    )


def run_bsc_component(
    rng: np.random.Generator,
    checks: np.ndarray,
    logicals: np.ndarray,
    p: float,
    runs: int,
    args: argparse.Namespace,
) -> dict[str, int | float]:
    if p <= 0.0:
        return {
            "logical_failures": 0,
            "decoder_syndrome_failures": 0,
            "runs": int(runs),
            "avg_weight": 0.0,
            "avg_correction_weight": 0.0,
            "bp_converged_runs": int(runs),
        }

    checks_u8 = checks.astype(np.uint8)
    logicals_u8 = logicals.astype(np.uint8)
    decoder = build_bposd_decoder(checks_u8, p, args)

    logical_failures = 0
    syndrome_failures = 0
    bp_converged = 0
    total_weight = 0
    total_correction_weight = 0
    n = checks_u8.shape[1]

    for _ in range(runs):
        error = (rng.random(n) < p).astype(np.uint8)
        syndrome = (checks_u8 @ error) % 2
        correction = decoder.decode(syndrome.astype(np.uint8)).astype(np.uint8) % 2
        residual = error ^ correction
        logical_failures += int(np.any((logicals_u8 @ residual) % 2))
        syndrome_failures += int(not np.array_equal((checks_u8 @ correction) % 2, syndrome))
        bp_converged += int(bool(getattr(decoder, "converge", False)))
        total_weight += int(error.sum())
        total_correction_weight += int(correction.sum())

    return {
        "logical_failures": int(logical_failures),
        "decoder_syndrome_failures": int(syndrome_failures),
        "runs": int(runs),
        "avg_weight": float(total_weight / runs),
        "avg_correction_weight": float(total_correction_weight / runs),
        "bp_converged_runs": int(bp_converged),
    }


def run_bsc_performance(
    code_path: Path,
    p_values: np.ndarray,
    runs: int,
    seed: int,
    confidence_z: float,
    args: argparse.Namespace,
) -> dict:
    code = load_code(code_path)
    if code.lx is None or code.lz is None:
        raise ValueError(f"{code_path} must contain Lx and Lz for BSC logical checks")

    rng = np.random.default_rng(seed)
    points = []
    for p in p_values:
        x_stats = run_bsc_component(rng, code.hz, code.lz, float(p), runs, args)
        z_stats = run_bsc_component(rng, code.hx, code.lx, float(p), runs, args)

        x_fail = int(x_stats["logical_failures"])
        z_fail = int(z_stats["logical_failures"])
        x_rate = x_fail / runs
        z_rate = z_fail / runs
        x_low, x_high = wilson_interval(x_fail, runs, confidence_z)
        z_low, z_high = wilson_interval(z_fail, runs, confidence_z)

        # If X and Z component errors are independent at the same p, this is the
        # induced full CSS logical-failure estimate from the two component curves.
        total_rate = 1.0 - (1.0 - x_rate) * (1.0 - z_rate)
        total_low = 1.0 - (1.0 - x_low) * (1.0 - z_low)
        total_high = 1.0 - (1.0 - x_high) * (1.0 - z_high)

        points.append(
            {
                "p": float(p),
                "runs_per_component": int(runs),
                "x_logical_failure": x_rate,
                "x_logical_failures": x_fail,
                "x_logical_failure_ci_low": x_low,
                "x_logical_failure_ci_high": x_high,
                "x_decoder_syndrome_failures": int(x_stats["decoder_syndrome_failures"]),
                "x_bp_converged_rate": float(x_stats["bp_converged_runs"] / runs),
                "x_avg_error_weight": float(x_stats["avg_weight"]),
                "x_avg_correction_weight": float(x_stats["avg_correction_weight"]),
                "z_logical_failure": z_rate,
                "z_logical_failures": z_fail,
                "z_logical_failure_ci_low": z_low,
                "z_logical_failure_ci_high": z_high,
                "z_decoder_syndrome_failures": int(z_stats["decoder_syndrome_failures"]),
                "z_bp_converged_rate": float(z_stats["bp_converged_runs"] / runs),
                "z_avg_error_weight": float(z_stats["avg_weight"]),
                "z_avg_correction_weight": float(z_stats["avg_correction_weight"]),
                "total_independent_logical_failure": total_rate,
                "total_independent_logical_failure_ci_low": total_low,
                "total_independent_logical_failure_ci_high": total_high,
            }
        )

    return {
        "code": code_metadata(code_path),
        "config": {
            "channel": "BSC component channel",
            "runs_per_component": int(runs),
            "seed": int(seed),
            "p_values": [float(p) for p in p_values],
            "confidence_z": float(confidence_z),
            "decoder": "ldpc.BpOsdDecoder",
            "bp_max_iter": int(args.bp_max_iter) if args.bp_max_iter > 0 else "n",
            "bp_method": args.bp_method,
            "ms_scaling_factor": float(args.ms_scaling_factor),
            "osd_method": args.osd_method,
            "osd_order": int(args.osd_order),
            "x_component": "decode X errors using Hz syndrome and Lz logical checks",
            "z_component": "decode Z errors using Hx syndrome and Lx logical checks",
        },
        "points": points,
    }


def plot_performance(
    result: dict,
    out_path: Path,
    *,
    channel: str,
    y_min: float = 1e-4,
) -> None:
    points = result["points"]
    p = np.asarray([point["p"] for point in points], dtype=float)
    code_name = result["code"]["name"]

    if channel == "bec":
        series = [
            ("x_ml_logical_failure", "X ML failure"),
            ("z_ml_logical_failure", "Z ML failure"),
            ("total_ml_logical_failure", "total ML failure"),
        ]
        xlabel = "erasure probability e"
        title = f"{code_name}: BEC performance"
    elif channel == "bsc":
        series = [
            ("x_logical_failure", "X BP+OSD failure"),
            ("z_logical_failure", "Z BP+OSD failure"),
            ("total_independent_logical_failure", "independent total"),
        ]
        xlabel = "BSC crossover probability p"
        title = f"{code_name}: BSC performance"
    else:
        raise ValueError(channel)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharex=True)
    for key, label in series:
        y = np.asarray([point[key] for point in points], dtype=float)
        low = np.asarray([point.get(f"{key}_ci_low", point.get(f"{key}_failure_ci_low", 0.0)) for point in points], dtype=float)
        high = np.asarray([point.get(f"{key}_ci_high", point.get(f"{key}_failure_ci_high", 0.0)) for point in points], dtype=float)
        for ax in axes:
            ax.plot(p, y, marker=".", linewidth=1.35, label=label)
            ax.fill_between(p, low, high, alpha=0.14)

    axes[0].set_ylabel("logical failure probability")
    axes[1].set_yscale("log")
    axes[1].set_ylim(bottom=y_min, top=1.0)
    axes[1].set_ylabel("logical failure probability")
    for ax in axes:
        ax.set_xlabel(xlabel)
        ax.set_ylim(top=1.02)
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[0].set_title("linear scale")
    axes[1].set_title("log scale")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_one_code(code_path: Path, args: argparse.Namespace, idx: int, total: int) -> None:
    stem = safe_stem(code_path)

    if args.channel in {"bec", "both"}:
        bec_dir = args.out_dir.resolve() / "bec"
        bec_json = bec_dir / f"{stem}_bec_performance.json"
        bec_csv = bec_dir / f"{stem}_bec_performance.csv"
        if args.skip_existing and bec_json.exists() and bec_csv.exists():
            bec_result = json.loads(bec_json.read_text(encoding="utf-8"))
        else:
            print(f"[{idx}/{total}] BEC performance {code_path.name}", flush=True)
            p_values = np.linspace(0.0, 1.0, args.bec_points)
            start = time.perf_counter()
            bec_result = run_bec_performance(
                code_path,
                p_values,
                args.bec_runs,
                seed=args.seed + idx * 1000 + 11,
                confidence_z=args.confidence_z,
            )
            bec_result["elapsed_seconds"] = float(time.perf_counter() - start)
            write_points(bec_result, bec_json, bec_csv)
        plot_performance(bec_result, bec_dir / f"{stem}_bec_performance.png", channel="bec")

    if args.channel in {"bsc", "both"}:
        bsc_dir = args.out_dir.resolve() / "bsc"
        bsc_json = bsc_dir / f"{stem}_bsc_performance.json"
        bsc_csv = bsc_dir / f"{stem}_bsc_performance.csv"
        if args.skip_existing and bsc_json.exists() and bsc_csv.exists():
            bsc_result = json.loads(bsc_json.read_text(encoding="utf-8"))
        else:
            print(f"[{idx}/{total}] BSC performance {code_path.name}", flush=True)
            p_values = np.linspace(0.0, args.bsc_p_max, args.bsc_points)
            start = time.perf_counter()
            bsc_result = run_bsc_performance(
                code_path,
                p_values,
                args.bsc_runs,
                seed=args.seed + idx * 1000 + 23,
                confidence_z=args.confidence_z,
                args=args,
            )
            bsc_result["elapsed_seconds"] = float(time.perf_counter() - start)
            write_points(bsc_result, bsc_json, bsc_csv)
        plot_performance(bsc_result, bsc_dir / f"{stem}_bsc_performance.png", channel="bsc")


def main() -> None:
    args = parse_args()
    codes = sorted(args.codes_dir.resolve().glob("*.npz"))
    if not codes:
        raise SystemExit(f"no .npz files found in {args.codes_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for idx, code_path in enumerate(codes, start=1):
        run_one_code(code_path, args, idx, len(codes))


if __name__ == "__main__":
    main()
