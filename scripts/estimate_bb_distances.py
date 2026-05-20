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


def exact_distance_milp(
    parity_check: np.ndarray,
    dual_logicals: np.ndarray,
    *,
    time_limit: float,
) -> dict[str, object]:
    """
    Solve one CSS distance side as a binary/integer linear program.

    For X distance, `parity_check` is H_Z and `dual_logicals` is L_Z. The
    constraints require H_Z x = 0 and L_Z x != 0 over GF(2), so every feasible
    solution is a nontrivial X logical. The Z side swaps X and Z.
    """
    h = np.asarray(parity_check, dtype=np.int8) % 2
    logicals = np.asarray(dual_logicals, dtype=np.int8) % 2
    checks, n_cols = h.shape
    k = logicals.shape[0]

    # Variables are x, parity slacks for H, logical syndrome bits y, and
    # parity slacks for L.  All are integer; x and y are binary.
    n_vars = n_cols + checks + k + k
    objective = np.zeros(n_vars)
    objective[:n_cols] = 1

    lower = np.zeros(n_vars)
    upper = np.full(n_vars, np.inf)
    upper[:n_cols] = 1
    upper[n_cols : n_cols + checks] = np.maximum(1, h.sum(axis=1) // 2)
    upper[n_cols + checks : n_cols + checks + k] = 1
    upper[n_cols + checks + k :] = np.maximum(1, logicals.sum(axis=1) // 2)

    rows = checks + k + 1
    constraints = lil_matrix((rows, n_vars), dtype=float)

    nz_rows, nz_cols = np.nonzero(h)
    for row, col in zip(nz_rows, nz_cols):
        constraints[row, col] = 1
    for row in range(checks):
        constraints[row, n_cols + row] = -2

    offset = checks
    nz_rows, nz_cols = np.nonzero(logicals)
    for row, col in zip(nz_rows, nz_cols):
        constraints[offset + row, col] = 1
    for row in range(k):
        constraints[offset + row, n_cols + checks + row] = -1
        constraints[offset + row, n_cols + checks + k + row] = -2

    for row in range(k):
        constraints[checks + k, n_cols + checks + row] = 1

    lower_bounds = np.r_[np.zeros(checks + k), [1]]
    upper_bounds = np.r_[np.zeros(checks + k), [np.inf]]
    result = milp(
        objective,
        integrality=np.ones(n_vars),
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(constraints.tocsr(), lower_bounds, upper_bounds),
        options={"time_limit": time_limit, "mip_rel_gap": 0},
    )

    value = None if result.fun is None else int(round(float(result.fun)))
    return {
        "value": value,
        "proved_optimal": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "mip_gap": None if getattr(result, "mip_gap", None) is None else float(result.mip_gap),
        "mip_dual_bound": None
        if getattr(result, "mip_dual_bound", None) is None
        else float(result.mip_dual_bound),
        "mip_node_count": None
        if getattr(result, "mip_node_count", None) is None
        else int(result.mip_node_count),
    }


def code_from_metadata(path: Path) -> BBCode:
    data = np.load(path, allow_pickle=True)
    metadata = json.loads(str(data["metadata_json"]))
    x, y = sp.symbols("x y")

    def polynomial(terms: list[list[int]]) -> sp.Expr:
        out = 0
        for row_shift, col_shift in terms:
            out += x**row_shift * y**col_shift
        return out

    return BBCode(
        {x: int(metadata["ell"]), y: int(metadata["m"])},
        polynomial(metadata["a_terms"]),
        polynomial(metadata["b_terms"]),
        field=2,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate sparse BB code distances.")
    parser.add_argument("codes", type=Path, nargs="+")
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT / "data" / "experiments" / "bb_sparse_code_parameters.csv",
    )
    parser.add_argument("--time-limit", type=float, default=120.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for path in args.codes:
        cache = load_exact_class_cache(path)
        code = code_from_metadata(path)
        print(f"CODE {path.name} n={cache.code.n} k={cache.code.k}", flush=True)

        t0 = time.perf_counter()
        lx = np.array(code.get_logical_ops(Pauli.X), dtype=np.uint8)
        lz = np.array(code.get_logical_ops(Pauli.Z), dtype=np.uint8)
        print(f"  logical_basis_seconds={time.perf_counter() - t0:.3f}", flush=True)

        side_results = {}
        for label, parity_check, dual_logicals in [
            ("x", np.array(code.matrix_z, dtype=np.uint8), lz),
            ("z", np.array(code.matrix_x, dtype=np.uint8), lx),
        ]:
            t0 = time.perf_counter()
            result = exact_distance_milp(
                parity_check,
                dual_logicals,
                time_limit=args.time_limit,
            )
            result["elapsed_seconds"] = time.perf_counter() - t0
            side_results[label] = result
            print(
                f"  {label}: value={result['value']} optimal={result['proved_optimal']} "
                f"gap={result['mip_gap']} elapsed={result['elapsed_seconds']:.3f}",
                flush=True,
            )

        values = [
            value
            for value in [side_results["x"]["value"], side_results["z"]["value"]]
            if value is not None
        ]
        rows.append(
            {
                "code": path.stem,
                "path": str(path),
                "n": cache.code.n,
                "k": cache.code.k,
                "rank_hx": cache.code.rank_hx,
                "rank_hz": cache.code.rank_hz,
                "d_upper_bound": min(values) if values else "",
                "d_x_upper_bound": side_results["x"]["value"] or "",
                "d_z_upper_bound": side_results["z"]["value"] or "",
                "d_x_proved_optimal": side_results["x"]["proved_optimal"],
                "d_z_proved_optimal": side_results["z"]["proved_optimal"],
                "d_x_mip_gap": side_results["x"]["mip_gap"],
                "d_z_mip_gap": side_results["z"]["mip_gap"],
                "distance_method": "MILP logical parity, upper bound unless proved_optimal",
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
