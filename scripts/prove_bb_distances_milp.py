from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exit_curve_experiments import load_exact_class_cache  # noqa: E402
from qldpc.codes import BBCode  # noqa: E402
from qldpc.objects import Pauli  # noqa: E402


def dense_from_sparse_npz(path: Path, prefix: str) -> np.ndarray:
    data = np.load(path, allow_pickle=True)
    matrix = np.zeros(
        (int(data[f"n_{prefix}"]), int(data[f"m_{prefix}"])),
        dtype=np.uint8,
    )
    matrix[data[f"rows_{prefix}"], data[f"cols_{prefix}"]] ^= 1
    return matrix


def bb_code_from_metadata(path: Path) -> BBCode:
    data = np.load(path, allow_pickle=True)
    metadata = json.loads(str(data["metadata_json"]))
    x, y = sp.symbols("x y")

    def polynomial(terms: list[list[int]]) -> sp.Expr:
        poly = 0
        for row_shift, col_shift in terms:
            poly += x**row_shift * y**col_shift
        return poly

    return BBCode(
        {x: int(metadata["ell"]), y: int(metadata["m"])},
        polynomial(metadata["a_terms"]),
        polynomial(metadata["b_terms"]),
        field=2,
    )


def load_distance_inputs(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    if "Hx" in data and "Hz" in data:
        if "Lx" not in data or "Lz" not in data:
            raise ValueError(f"{path} must contain Lx/Lz for distance proving")
        return {
            "hx": np.asarray(data["Hx"], dtype=np.uint8) % 2,
            "hz": np.asarray(data["Hz"], dtype=np.uint8) % 2,
            "lx": np.asarray(data["Lx"], dtype=np.uint8) % 2,
            "lz": np.asarray(data["Lz"], dtype=np.uint8) % 2,
        }

    if "metadata_json" not in data:
        raise ValueError(f"{path} is sparse but has no BB metadata_json")
    code = bb_code_from_metadata(path)
    return {
        "hx": np.array(code.matrix_x, dtype=np.uint8),
        "hz": np.array(code.matrix_z, dtype=np.uint8),
        "lx": np.array(code.get_logical_ops(Pauli.X), dtype=np.uint8),
        "lz": np.array(code.get_logical_ops(Pauli.Z), dtype=np.uint8),
    }


def one_logical_feasibility_milp(
    parity_check: np.ndarray,
    logical_row: np.ndarray,
    max_weight: int,
    *,
    time_limit: float,
) -> dict[str, object]:
    """
    Decide whether there is x with H x = 0, logical_row x = 1, wt(x)<=max_weight.

    If every logical basis row is infeasible for max_weight=d-1, then no
    nontrivial logical operator of that side has weight below d.
    """
    h = np.asarray(parity_check, dtype=np.int8) % 2
    logical = np.asarray(logical_row, dtype=np.int8).reshape(-1) % 2
    checks, n_cols = h.shape

    n_vars = n_cols + checks + 1
    objective = np.zeros(n_vars)
    integrality = np.ones(n_vars)

    lower = np.zeros(n_vars)
    upper = np.full(n_vars, np.inf)
    upper[:n_cols] = 1
    upper[n_cols : n_cols + checks] = np.maximum(1, h.sum(axis=1) // 2)
    upper[-1] = max(1, int(logical.sum() // 2))

    constraints = lil_matrix((checks + 2, n_vars), dtype=float)
    nz_rows, nz_cols = np.nonzero(h)
    for row, col in zip(nz_rows, nz_cols):
        constraints[row, col] = 1
    for row in range(checks):
        constraints[row, n_cols + row] = -2

    logical_nz = np.flatnonzero(logical)
    for col in logical_nz:
        constraints[checks, int(col)] = 1
    constraints[checks, -1] = -2

    for col in range(n_cols):
        constraints[checks + 1, col] = 1

    lower_bounds = np.r_[np.zeros(checks), [1, 0]]
    upper_bounds = np.r_[np.zeros(checks), [1, max_weight]]

    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(constraints.tocsr(), lower_bounds, upper_bounds),
        options={"time_limit": time_limit, "mip_rel_gap": 0},
    )

    weight = None
    if result.x is not None:
        weight = int(round(float(np.sum(result.x[:n_cols]))))
    return {
        "feasible": result.success and result.x is not None,
        "infeasible": int(result.status) == 2,
        "status": int(result.status),
        "message": str(result.message),
        "weight": weight,
        "mip_gap": None if getattr(result, "mip_gap", None) is None else float(result.mip_gap),
        "mip_node_count": None
        if getattr(result, "mip_node_count", None) is None
        else int(result.mip_node_count),
    }


def prove_side(
    parity_check: np.ndarray,
    dual_logicals: np.ndarray,
    candidate: int,
    *,
    time_limit_per_row: float,
) -> dict[str, object]:
    max_weight = int(candidate) - 1
    row_results = []
    t0 = time.perf_counter()
    for row_idx, logical_row in enumerate(dual_logicals):
        row_t0 = time.perf_counter()
        result = one_logical_feasibility_milp(
            parity_check,
            logical_row,
            max_weight,
            time_limit=time_limit_per_row,
        )
        result["row"] = row_idx
        result["elapsed_seconds"] = time.perf_counter() - row_t0
        row_results.append(result)
        print(
            f"    row={row_idx} feasible={result['feasible']} "
            f"infeasible={result['infeasible']} status={result['status']} "
            f"weight={result['weight']} elapsed={result['elapsed_seconds']:.3f}",
            flush=True,
        )
        if result["feasible"]:
            return {
                "proved_exact": False,
                "candidate": candidate,
                "found_below_candidate": result["weight"],
                "checked_rows": row_results,
                "elapsed_seconds": time.perf_counter() - t0,
            }

    proved = all(result["infeasible"] for result in row_results)
    return {
        "proved_exact": proved,
        "candidate": candidate,
        "found_below_candidate": "",
        "checked_rows": row_results,
        "elapsed_seconds": time.perf_counter() - t0,
    }


def read_parameter_candidates(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["code"]: row for row in csv.DictReader(f)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Try to prove BB distances exactly via MILP.")
    parser.add_argument("codes", type=Path, nargs="+")
    parser.add_argument(
        "--params-csv",
        type=Path,
        default=ROOT / "data" / "experiments" / "bb_code_parameters.csv",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT / "data" / "experiments" / "bb_code_distance_proofs.csv",
    )
    parser.add_argument("--time-limit-per-row", type=float, default=300.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = read_parameter_candidates(args.params_csv)
    output_rows: list[dict[str, object]] = []

    for path in args.codes:
        code_name = path.stem
        if code_name not in candidates:
            raise SystemExit(f"{code_name} not found in {args.params_csv}")
        candidate_row = candidates[code_name]
        inputs = load_distance_inputs(path)
        cache = load_exact_class_cache(path)
        print(f"CODE {code_name} n={cache.code.n} k={cache.code.k}", flush=True)

        side_data = {
            "x": (inputs["hz"], inputs["lz"], int(candidate_row["d_x_recorded"])),
            "z": (inputs["hx"], inputs["lx"], int(candidate_row["d_z_recorded"])),
        }
        proof_results = {}
        for side, (parity_check, dual_logicals, candidate) in side_data.items():
            print(f"  SIDE {side} candidate={candidate}", flush=True)
            proof_results[side] = prove_side(
                parity_check,
                dual_logicals,
                candidate,
                time_limit_per_row=args.time_limit_per_row,
            )
            print(
                f"  SIDE {side} proved_exact={proof_results[side]['proved_exact']} "
                f"elapsed={proof_results[side]['elapsed_seconds']:.3f}",
                flush=True,
            )

        exact_sides = [
            side
            for side, result in proof_results.items()
            if result["proved_exact"]
        ]
        output_rows.append(
            {
                "code": code_name,
                "path": str(path),
                "n": cache.code.n,
                "k": cache.code.k,
                "d_x_recorded": candidate_row["d_x_recorded"],
                "d_z_recorded": candidate_row["d_z_recorded"],
                "d_recorded": candidate_row["d_recorded"],
                "d_x_proved_exact": proof_results["x"]["proved_exact"],
                "d_z_proved_exact": proof_results["z"]["proved_exact"],
                "d_proved_exact": len(exact_sides) == 2
                and min(int(candidate_row["d_x_recorded"]), int(candidate_row["d_z_recorded"]))
                == int(candidate_row["d_recorded"]),
                "x_found_below_candidate": proof_results["x"]["found_below_candidate"],
                "z_found_below_candidate": proof_results["z"]["found_below_candidate"],
                "time_limit_per_row": args.time_limit_per_row,
                "method": "MILP infeasibility for each dual logical basis row below recorded candidate",
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"wrote {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
