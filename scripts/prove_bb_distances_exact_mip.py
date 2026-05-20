from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import sympy as sp
from mip import BINARY, Model, OptimizationStatus, minimize, xsum


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exit_curve_experiments import load_exact_class_cache  # noqa: E402
from qldpc.codes import BBCode  # noqa: E402
from qldpc.objects import Pauli  # noqa: E402


def as_dense_binary(matrix: object) -> np.ndarray:
    """Convert dense, scipy sparse, or qLDPC matrix-like objects to uint8 GF(2)."""
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.uint8) % 2


def load_bb_from_metadata(path: Path) -> BBCode:
    data = np.load(path, allow_pickle=True)
    metadata = json.loads(str(data["metadata_json"]))
    x, y = sp.symbols("x y")

    def polynomial(terms: Iterable[Iterable[int]]) -> sp.Expr:
        poly = 0
        for row_shift, col_shift in terms:
            poly += x ** int(row_shift) * y ** int(col_shift)
        return poly

    return BBCode(
        {x: int(metadata["ell"]), y: int(metadata["m"])},
        polynomial(metadata["a_terms"]),
        polynomial(metadata["b_terms"]),
        field=2,
    )


def load_distance_inputs(path: Path) -> dict[str, np.ndarray]:
    """
    Return H_X, H_Z, L_X, L_Z for dense legacy codes or sparse generated BB codes.

    Sparse files store only the checks and metadata; qLDPC is used to reconstruct a
    canonical logical basis. This is deliberately done lazily so first-use dense
    files still work without any precomputed sparse sidecar data.
    """
    data = np.load(path, allow_pickle=True)
    if "Hx" in data and "Hz" in data:
        if "Lx" not in data or "Lz" not in data:
            raise ValueError(f"{path} must contain Lx and Lz for exact distance proving")
        return {
            "hx": as_dense_binary(data["Hx"]),
            "hz": as_dense_binary(data["Hz"]),
            "lx": as_dense_binary(data["Lx"]),
            "lz": as_dense_binary(data["Lz"]),
        }

    if "metadata_json" not in data:
        raise ValueError(f"{path} is sparse but has no metadata_json")

    code = load_bb_from_metadata(path)
    return {
        "hx": as_dense_binary(code.matrix_x),
        "hz": as_dense_binary(code.matrix_z),
        "lx": as_dense_binary(code.get_logical_ops(Pauli.X)),
        "lz": as_dense_binary(code.get_logical_ops(Pauli.Z)),
    }


def parity_slack_bits(max_row_weight: int) -> int:
    """Number of binary slack bits needed for sum(row support) = 2 * slack."""
    max_slack = max(1, int(max_row_weight) // 2)
    return max(1, math.ceil(math.log2(max_slack + 1)))


def solve_logical_row_distance(
    parity_check: np.ndarray,
    logical_row: np.ndarray,
    *,
    time_limit_seconds: float,
    verbose: bool,
) -> dict[str, object]:
    """
    Minimize wt(x) with H x = 0 mod 2 and logical_row x = 1 mod 2.

    Solving this once per dual logical basis row gives the exact CSS distance
    side when every row solve is certified optimal.
    """
    h = as_dense_binary(parity_check)
    logical = as_dense_binary(logical_row).reshape(-1)
    checks, n_cols = h.shape

    check_bits = parity_slack_bits(int(h.sum(axis=1).max(initial=0)))
    logical_bits = parity_slack_bits(int(logical.sum(initial=0)))
    n_vars = n_cols + checks * check_bits + logical_bits

    model = Model(sense="MIN", solver_name="CBC")
    model.verbose = int(verbose)
    model.max_seconds = float(time_limit_seconds)
    variables = [model.add_var(var_type=BINARY) for _ in range(n_vars)]
    model.objective = minimize(xsum(variables[col] for col in range(n_cols)))

    for row in range(checks):
        support = np.flatnonzero(h[row])
        slack_offset = n_cols + row * check_bits
        model += (
            xsum(variables[int(col)] for col in support)
            - xsum((2 * (1 << bit)) * variables[slack_offset + bit] for bit in range(check_bits))
            == 0
        )

    logical_support = np.flatnonzero(logical)
    logical_offset = n_cols + checks * check_bits
    model += (
        xsum(variables[int(col)] for col in logical_support)
        - xsum(
            (2 * (1 << bit)) * variables[logical_offset + bit]
            for bit in range(logical_bits)
        )
        == 1
    )

    start = time.perf_counter()
    status = model.optimize()
    elapsed = time.perf_counter() - start

    value = None
    if model.num_solutions:
        value = int(round(float(model.objective_value)))

    bound = model.objective_bound
    if bound is not None:
        bound = float(bound)

    gap = model.gap
    if gap is not None and np.isfinite(gap):
        gap = float(gap)
    else:
        gap = None

    return {
        "status": str(status).replace("OptimizationStatus.", ""),
        "is_optimal": status == OptimizationStatus.OPTIMAL,
        "value": "" if value is None else value,
        "lower_bound": "" if bound is None else bound,
        "gap": "" if gap is None else gap,
        "elapsed_seconds": elapsed,
        "num_solutions": int(model.num_solutions),
        "check_slack_bits": check_bits,
        "logical_slack_bits": logical_bits,
    }


def load_completed_rows(row_csv: Path) -> set[tuple[str, str, int]]:
    if not row_csv.exists():
        return set()
    completed: set[tuple[str, str, int]] = set()
    with row_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("is_optimal") == "True":
                completed.add((row["code"], row["side"], int(row["logical_row"])))
    return completed


def append_row(row_csv: Path, row: dict[str, object]) -> None:
    fieldnames = [
        "code",
        "path",
        "side",
        "logical_row",
        "status",
        "is_optimal",
        "value",
        "lower_bound",
        "gap",
        "elapsed_seconds",
        "num_solutions",
        "check_slack_bits",
        "logical_slack_bits",
        "time_limit_seconds",
    ]
    exists = row_csv.exists()
    row_csv.parent.mkdir(parents=True, exist_ok=True)
    with row_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def summarize_rows(row_csv: Path, summary_csv: Path, assume_xz_symmetric: bool) -> None:
    grouped: dict[tuple[str, str], dict[int, dict[str, str]]] = {}
    metadata: dict[str, dict[str, str]] = {}
    with row_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["code"], row["side"])
            logical_row = int(row["logical_row"])
            rows_for_side = grouped.setdefault(key, {})
            previous = rows_for_side.get(logical_row)
            if previous is None or (
                previous.get("is_optimal") != "True" and row.get("is_optimal") == "True"
            ):
                rows_for_side[logical_row] = row
            elif previous.get("is_optimal") == row.get("is_optimal"):
                # Keep the latest same-quality attempt, which usually has the longer limit.
                rows_for_side[logical_row] = row
            metadata.setdefault(row["code"], {"path": row["path"]})

    side_summaries: dict[tuple[str, str], dict[str, object]] = {}
    for key, rows_by_logical in grouped.items():
        rows = list(rows_by_logical.values())
        optimal_rows = [row for row in rows if row["is_optimal"] == "True" and row["value"] != ""]
        values = [int(float(row["value"])) for row in optimal_rows]
        side_summaries[key] = {
            "rows_completed": len(optimal_rows),
            "rows_total": len(rows),
            "side_distance": min(values) if values else "",
            "side_proved_exact": len(optimal_rows) == len(rows) and bool(rows),
            "side_elapsed_seconds": sum(float(row["elapsed_seconds"]) for row in rows),
        }

    output_rows: list[dict[str, object]] = []
    for code in sorted(metadata):
        sides = sorted(side for row_code, side in side_summaries if row_code == code)
        side_distance_values = [
            side_summaries[(code, side)]["side_distance"]
            for side in sides
            if side_summaries[(code, side)]["side_distance"] != ""
        ]
        side_proofs = [
            bool(side_summaries[(code, side)]["side_proved_exact"])
            for side in sides
        ]
        proved_exact = bool(side_proofs) and all(side_proofs)
        if assume_xz_symmetric and side_proofs:
            proved_exact = any(side_proofs)
        output_rows.append(
            {
                "code": code,
                "path": metadata[code]["path"],
                "sides_run": ";".join(sides),
                "d_exact_or_bound": min(side_distance_values) if side_distance_values else "",
                "proved_exact": proved_exact,
                "assume_xz_symmetric": assume_xz_symmetric,
                "side_details_json": json.dumps(
                    {side: side_summaries[(code, side)] for side in sides},
                    sort_keys=True,
                ),
            }
        )

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prove CSS BB distances exactly with row-wise CBC/MIP solves."
    )
    parser.add_argument("codes", type=Path, nargs="+")
    parser.add_argument(
        "--side",
        choices=["x", "z"],
        action="append",
        default=None,
        help="CSS distance side to prove. Repeat for both. Default: z.",
    )
    parser.add_argument("--time-limit-per-row", type=float, default=600.0)
    parser.add_argument(
        "--row-csv",
        type=Path,
        default=ROOT / "data" / "experiments" / "bb_exact_distance_rows.csv",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=ROOT / "data" / "experiments" / "bb_exact_distance_summary.csv",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--assume-bb-xz-symmetric", action="store_true")
    parser.add_argument("--verbose-solver", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sides = args.side or ["z"]
    completed = load_completed_rows(args.row_csv) if args.resume else set()

    for path in args.codes:
        cache = load_exact_class_cache(path)
        code_name = path.stem
        print(f"CODE {code_name} n={cache.code.n} k={cache.code.k}", flush=True)
        inputs = load_distance_inputs(path)
        side_inputs = {
            "x": (inputs["hz"], inputs["lz"]),
            "z": (inputs["hx"], inputs["lx"]),
        }

        for side in sides:
            parity_check, dual_logicals = side_inputs[side]
            print(f"  SIDE {side}: {len(dual_logicals)} logical rows", flush=True)
            for row_idx, logical_row in enumerate(dual_logicals):
                key = (code_name, side, row_idx)
                if key in completed:
                    print(f"    row {row_idx}: already optimal, skipping", flush=True)
                    continue
                start = time.perf_counter()
                result = solve_logical_row_distance(
                    parity_check,
                    logical_row,
                    time_limit_seconds=args.time_limit_per_row,
                    verbose=args.verbose_solver,
                )
                row = {
                    "code": code_name,
                    "path": str(path),
                    "side": side,
                    "logical_row": row_idx,
                    "time_limit_seconds": args.time_limit_per_row,
                    **result,
                }
                append_row(args.row_csv, row)
                print(
                    f"    row {row_idx}: status={row['status']} value={row['value']} "
                    f"bound={row['lower_bound']} elapsed={time.perf_counter() - start:.1f}s",
                    flush=True,
                )

    summarize_rows(args.row_csv, args.summary_csv, args.assume_bb_xz_symmetric)
    print(f"wrote {args.row_csv}", flush=True)
    print(f"wrote {args.summary_csv}", flush=True)


if __name__ == "__main__":
    main()
