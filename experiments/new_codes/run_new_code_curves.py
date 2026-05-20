from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.bsc_gexit_class_mcmc import compute_result as compute_bsc_result
from experiments.bsc_gexit_class_mcmc import write_outputs as write_bsc_outputs
from experiments.exit_curve_experiments import (
    centered_ps,
    rank_gf2,
    run_code,
    run_paired_exact_class_target_code,
)
from experiments.exit_curve_experiments import load_code


DEFAULT_CODES_DIR = REPO_ROOT / "codes" / "new"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "experiments" / "new_codes"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate new qLDPC code files and generate EXIT/GEXIT curves."
    )
    parser.add_argument("--codes-dir", type=Path, default=DEFAULT_CODES_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--exit-runs", type=int, default=500)
    parser.add_argument("--exit-points", type=int, default=21)
    parser.add_argument("--paired-runs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--gexit-points", type=int, default=21)
    parser.add_argument("--gexit-runs", type=int, default=20)
    parser.add_argument("--gexit-samples", type=int, default=2000)
    parser.add_argument("--gexit-burnin", type=int, default=500)
    parser.add_argument("--gexit-thin", type=int, default=1)
    return parser.parse_args()


def safe_stem(path: Path) -> str:
    stem = path.stem.replace("[[", "").replace("]]", "")
    return re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")


def binary_matrix_product_weight(left: np.ndarray, right: np.ndarray) -> int:
    return int(np.count_nonzero((left.astype(np.uint8) @ right.astype(np.uint8).T) % 2))


def row_weight_stats(matrix: np.ndarray) -> tuple[int, float, int]:
    weights = np.sum(matrix % 2, axis=1)
    if not weights.size:
        return 0, 0.0, 0
    return int(weights.min()), float(weights.mean()), int(weights.max())


def col_weight_stats(matrix: np.ndarray) -> tuple[int, float, int]:
    weights = np.sum(matrix % 2, axis=0)
    if not weights.size:
        return 0, 0.0, 0
    return int(weights.min()), float(weights.mean()), int(weights.max())


def validate_code(path: Path) -> dict[str, object]:
    code = load_code(path)
    hx = code.hx.astype(np.uint8) % 2
    hz = code.hz.astype(np.uint8) % 2
    lx = code.lx.astype(np.uint8) % 2 if code.lx is not None else None
    lz = code.lz.astype(np.uint8) % 2 if code.lz is not None else None

    match = re.search(r"\[\[(\d+),(\d+),(\d+)\]\]", path.name)
    filename_n = filename_k = filename_d = None
    if match:
        filename_n, filename_k, filename_d = (int(value) for value in match.groups())

    issues: list[str] = []
    warnings: list[str] = []
    if filename_n is not None and filename_n != code.n:
        issues.append(f"filename n={filename_n} but computed n={code.n}")
    if filename_k is not None and filename_k != code.k:
        issues.append(f"filename k={filename_k} but computed k={code.k}")

    comm_entries = binary_matrix_product_weight(hx, hz)
    if comm_entries:
        issues.append(f"Hx Hz^T has {comm_entries} nonzero entries")

    lx_hz_entries = None
    lz_hx_entries = None
    lx_rank_increment = None
    lz_rank_increment = None
    pair_rank = None
    pair_is_identity = None
    if lx is None or lz is None:
        issues.append("missing Lx or Lz")
    else:
        lx_hz_entries = binary_matrix_product_weight(lx, hz)
        lz_hx_entries = binary_matrix_product_weight(lz, hx)
        if lx_hz_entries:
            issues.append(f"Lx Hz^T has {lx_hz_entries} nonzero entries")
        if lz_hx_entries:
            issues.append(f"Lz Hx^T has {lz_hx_entries} nonzero entries")

        lx_rank_increment = rank_gf2(np.vstack([hx, lx])) - rank_gf2(hx)
        lz_rank_increment = rank_gf2(np.vstack([hz, lz])) - rank_gf2(hz)
        if lx_rank_increment != code.k:
            issues.append(f"Lx rank increment is {lx_rank_increment}, expected k={code.k}")
        if lz_rank_increment != code.k:
            issues.append(f"Lz rank increment is {lz_rank_increment}, expected k={code.k}")

        pair_matrix = (lx @ lz.T) % 2
        pair_rank = rank_gf2(pair_matrix)
        pair_is_identity = bool(
            pair_matrix.shape == (code.k, code.k)
            and np.array_equal(pair_matrix, np.eye(code.k, dtype=np.uint8))
        )
        if pair_rank != code.k:
            issues.append(f"Lx Lz^T rank is {pair_rank}, expected k={code.k}")
        elif not pair_is_identity:
            warnings.append("Lx Lz^T is full rank but not the identity pairing")

    hx_rw_min, hx_rw_mean, hx_rw_max = row_weight_stats(hx)
    hz_rw_min, hz_rw_mean, hz_rw_max = row_weight_stats(hz)
    hx_cw_min, hx_cw_mean, hx_cw_max = col_weight_stats(hx)
    hz_cw_min, hz_cw_mean, hz_cw_max = col_weight_stats(hz)

    return {
        "file": path.name,
        "path": str(path),
        "n": code.n,
        "k": code.k,
        "filename_n": filename_n,
        "filename_k": filename_k,
        "filename_d": filename_d,
        "rank_hx": code.rank_hx,
        "rank_hz": code.rank_hz,
        "num_hx_rows": int(hx.shape[0]),
        "num_hz_rows": int(hz.shape[0]),
        "hx_hz_noncommuting_entries": comm_entries,
        "lx_hz_noncommuting_entries": lx_hz_entries,
        "lz_hx_noncommuting_entries": lz_hx_entries,
        "lx_rank_increment": lx_rank_increment,
        "lz_rank_increment": lz_rank_increment,
        "lx_lz_pair_rank": pair_rank,
        "lx_lz_pair_is_identity": pair_is_identity,
        "hx_row_weight_min": hx_rw_min,
        "hx_row_weight_mean": hx_rw_mean,
        "hx_row_weight_max": hx_rw_max,
        "hz_row_weight_min": hz_rw_min,
        "hz_row_weight_mean": hz_rw_mean,
        "hz_row_weight_max": hz_rw_max,
        "hx_col_weight_min": hx_cw_min,
        "hx_col_weight_mean": hx_cw_mean,
        "hx_col_weight_max": hx_cw_max,
        "hz_col_weight_min": hz_cw_min,
        "hz_col_weight_mean": hz_cw_mean,
        "hz_col_weight_max": hz_cw_max,
        "status": "ok" if not issues else "issue",
        "issues": "; ".join(issues),
        "warnings": "; ".join(warnings),
    }


def write_table(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_result(result: dict, json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    fields = sorted({key for point in result["points"] for key in point})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["points"])


def plot_erasure_bit_scale(result: dict, out_path: Path, component: str) -> None:
    points = result["points"]
    p = np.asarray([point["p"] for point in points], dtype=float)
    code = result["code"]
    prefix = f"exact_{component}"
    labels = {
        "error": rf"$\mathbb{{E}}H({component}\mid S)$",
        "class": rf"$\mathbb{{E}}H(C_{component.upper()}\mid S)$",
        "saved_by_stabilizers": rf"$\mathbb{{E}}H({component}\mid C_{component.upper()},S)$",
    }
    series = [
        (f"{prefix}_error", labels["error"]),
        (f"{prefix}_class", labels["class"]),
        (f"{prefix}_saved_by_stabilizers", labels["saved_by_stabilizers"]),
    ]
    fig, ax = plt.subplots(figsize=(6.1, 4.1))
    for key, label in series:
        ax.plot(p, [point[key] for point in points], marker=".", linewidth=1.35, label=label)
    ax.axhline(code["k"], color="black", linestyle="--", linewidth=0.9, label=r"$k$")
    ax.set_xlabel("erasure probability e")
    ax.set_ylabel("expected entropy (bits)")
    ax.set_title(f"{code['name']}: erasure {component.upper()} bit scale")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_erasure_derivative(result: dict, out_path: Path, component: str) -> None:
    points = result["points"]
    p = np.asarray([point["p"] for point in points], dtype=float)
    key = f"exact_{component}_class_component_norm_dp"
    y = np.asarray([point.get(key, 0.0) for point in points], dtype=float)
    yerr_key = f"{key}_stderr"
    fig, ax = plt.subplots(figsize=(6.1, 4.1))
    if yerr_key in points[0]:
        yerr = np.asarray([point[yerr_key] for point in points], dtype=float)
        ax.errorbar(p, y, yerr=yerr, marker=".", linewidth=1.35, capsize=2.0)
    else:
        ax.plot(p, y, marker=".", linewidth=1.35)
    ax.set_xlabel("erasure probability e")
    ax.set_ylabel(rf"$d[H(C_{component.upper()}\mid E,S)/n]/de$")
    ax.set_title(f"{result['code']['name']}: erasure {component.upper()} EXIT derivative")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_exit_jobs(
    code_path: Path,
    out_dir: Path,
    *,
    runs: int,
    points: int,
    paired_runs: int,
    seed: int,
    skip_existing: bool,
) -> None:
    stem = safe_stem(code_path)
    bit_json = out_dir / f"{stem}_erasure_bit_scale.json"
    bit_csv = out_dir / f"{stem}_erasure_bit_scale.csv"
    if not skip_existing or not bit_json.exists() or not bit_csv.exists():
        ps = np.linspace(0.0, 1.0, points)
        bit_result = run_code(
            code_path,
            ps,
            runs,
            seed=seed,
            include_exact=True,
            include_peeling=False,
        )
        write_result(bit_result, bit_json, bit_csv)
    else:
        bit_result = json.loads(bit_json.read_text(encoding="utf-8"))

    plot_erasure_bit_scale(bit_result, out_dir / f"{stem}_erasure_x_bit_scale.png", "x")
    plot_erasure_bit_scale(bit_result, out_dir / f"{stem}_erasure_z_bit_scale.png", "z")

    paired_json = out_dir / f"{stem}_erasure_class_paired_derivative.json"
    paired_csv = out_dir / f"{stem}_erasure_class_paired_derivative.csv"
    if not skip_existing or not paired_json.exists() or not paired_csv.exists():
        ps = centered_ps(edge_step=0.1, shoulder_step=0.05, center_step=0.02)
        paired_result = run_paired_exact_class_target_code(
            code_path,
            ps,
            paired_runs,
            seed=seed,
            component="both",
            use_logicals=True,
        )
        write_result(paired_result, paired_json, paired_csv)
    else:
        paired_result = json.loads(paired_json.read_text(encoding="utf-8"))

    plot_erasure_derivative(
        paired_result,
        out_dir / f"{stem}_erasure_x_exit_derivative.png",
        "x",
    )
    plot_erasure_derivative(
        paired_result,
        out_dir / f"{stem}_erasure_z_exit_derivative.png",
        "z",
    )


def run_gexit_jobs(
    code_path: Path,
    out_dir: Path,
    *,
    points: int,
    runs: int,
    samples: int,
    burnin: int,
    thin: int,
    seed: int,
    skip_existing: bool,
) -> None:
    tikz_dir = out_dir / "tikz"
    for component, seed_offset in (("x", 101), ("z", 202)):
        stem = safe_stem(code_path)
        expected = out_dir / f"{stem}_bsc_{component}_class_mcmc.json"
        if skip_existing and expected.exists():
            continue
        args = SimpleNamespace(
            code=code_path,
            component=component,
            points=points,
            runs=runs,
            approx_samples=samples,
            mcmc_burnin=burnin,
            mcmc_thin=thin,
            seed=seed + seed_offset,
        )
        result = compute_bsc_result(args)
        write_bsc_outputs(result, out_dir, tikz_dir)


def main() -> None:
    args = parse_args()
    codes = sorted(args.codes_dir.resolve().glob("*.npz"))
    if not codes:
        raise SystemExit(f"no .npz files found in {args.codes_dir}")

    validation_rows = [validate_code(path) for path in codes]
    validation_dir = args.out_dir.resolve() / "validation"
    write_table(validation_rows, validation_dir / "new_code_validation.csv")
    (validation_dir / "new_code_validation.json").write_text(
        json.dumps(validation_rows, indent=2),
        encoding="utf-8",
    )
    print(f"validated {len(validation_rows)} codes into {validation_dir}")

    if args.validate_only:
        return

    exit_dir = args.out_dir.resolve() / "exit_curves"
    gexit_dir = args.out_dir.resolve() / "gexit_curves"
    for idx, code_path in enumerate(codes, start=1):
        print(f"[{idx}/{len(codes)}] EXIT {code_path.name}", flush=True)
        run_exit_jobs(
            code_path,
            exit_dir,
            runs=args.exit_runs,
            points=args.exit_points,
            paired_runs=args.paired_runs,
            seed=args.seed + idx * 1000,
            skip_existing=args.skip_existing,
        )
        print(f"[{idx}/{len(codes)}] BSC GEXIT {code_path.name}", flush=True)
        run_gexit_jobs(
            code_path,
            gexit_dir,
            points=args.gexit_points,
            runs=args.gexit_runs,
            samples=args.gexit_samples,
            burnin=args.gexit_burnin,
            thin=args.gexit_thin,
            seed=args.seed + idx * 1000,
            skip_existing=args.skip_existing,
        )


if __name__ == "__main__":
    main()
