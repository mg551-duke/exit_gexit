from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


FAMILIES: dict[str, dict[str, object]] = {
    # The existing gross_HxHzLxLz.npz file is the ell=12, m=6 member of
    # this BB family: A = y + y^2 + x^3 and B = y^3 + x + x^2.
    "gross": {
        "a_terms": [(0, 1), (0, 2), (3, 0)],
        "b_terms": [(0, 3), (1, 0), (2, 0)],
    },
    # The existing two_gross_HxHzLxLz.npz file is the ell=12, m=12 member
    # of this variant: A = y^2 + y^7 + x^3 and B = y^3 + x + x^2.
    "two_gross": {
        "a_terms": [(0, 2), (0, 7), (3, 0)],
        "b_terms": [(0, 3), (1, 0), (2, 0)],
    },
    # The existing B1_HxHzLxLz.npz file is the ell=21, m=21 member of this
    # family. It is included for completeness, but scaled copies can have
    # k=0 for some choices of ell and m.
    "B1": {
        "a_terms": [(1, 15), (15, 0), (18, 9)],
        "b_terms": [(0, 0), (2, 15), (2, 20)],
    },
}


def torus_index(row: int, col: int, ell: int, m: int) -> int:
    """Map a coordinate on the ell-by-m torus to one matrix column."""
    return (row % ell) * m + (col % m)


def add_row_terms(
    rows: list[int],
    cols: list[int],
    row_idx: int,
    base_col: int,
    row_coord: int,
    col_coord: int,
    terms: list[tuple[int, int]],
    ell: int,
    m: int,
    *,
    transpose: bool = False,
) -> None:
    """
    Add one sparse row for a circulant polynomial.

    For A and B in H_X, terms are forward shifts. For H_Z=[B^T,A^T], the
    transpose of a shift is the inverse shift, so `transpose=True` subtracts
    the exponents instead.
    """
    row_cols: set[int] = set()
    for d_row, d_col in terms:
        if transpose:
            d_row = -d_row
            d_col = -d_col
        row_cols.add(base_col + torus_index(row_coord + d_row, col_coord + d_col, ell, m))

    for col_idx in sorted(row_cols):
        rows.append(row_idx)
        cols.append(col_idx)


def bivariate_bicycle_sparse(
    ell: int,
    m: int,
    a_terms: list[tuple[int, int]],
    b_terms: list[tuple[int, int]],
) -> dict[str, np.ndarray]:
    """
    Build H_X=[A,B] and H_Z=[B^T,A^T] directly as sparse COO arrays.

    There are N=ell*m rows in each parity-check matrix and 2N qubits. Each
    row has weight six unless two polynomial shifts collide modulo ell or m.
    """
    n_rows = ell * m
    n_cols = 2 * n_rows

    hx_rows: list[int] = []
    hx_cols: list[int] = []
    hz_rows: list[int] = []
    hz_cols: list[int] = []

    for row_coord in range(ell):
        for col_coord in range(m):
            row_idx = row_coord * m + col_coord
            add_row_terms(
                hx_rows,
                hx_cols,
                row_idx,
                0,
                row_coord,
                col_coord,
                a_terms,
                ell,
                m,
            )
            add_row_terms(
                hx_rows,
                hx_cols,
                row_idx,
                n_rows,
                row_coord,
                col_coord,
                b_terms,
                ell,
                m,
            )
            add_row_terms(
                hz_rows,
                hz_cols,
                row_idx,
                0,
                row_coord,
                col_coord,
                b_terms,
                ell,
                m,
                transpose=True,
            )
            add_row_terms(
                hz_rows,
                hz_cols,
                row_idx,
                n_rows,
                row_coord,
                col_coord,
                a_terms,
                ell,
                m,
                transpose=True,
            )

    return {
        "rows_HX": np.array(hx_rows, dtype=np.int64),
        "cols_HX": np.array(hx_cols, dtype=np.int64),
        "n_HX": np.array(n_rows, dtype=np.int64),
        "m_HX": np.array(n_cols, dtype=np.int64),
        "rows_HZ": np.array(hz_rows, dtype=np.int64),
        "cols_HZ": np.array(hz_cols, dtype=np.int64),
        "n_HZ": np.array(n_rows, dtype=np.int64),
        "m_HZ": np.array(n_cols, dtype=np.int64),
    }


def parse_term(value: str) -> tuple[int, int]:
    left, right = value.split(",", maxsplit=1)
    return int(left), int(right)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate sparse bivariate-bicycle CSS code matrices."
    )
    parser.add_argument("--family", choices=sorted(FAMILIES), default="gross")
    parser.add_argument("--ell", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument(
        "--a-term",
        type=parse_term,
        action="append",
        default=None,
        help="Override family A term as row_shift,col_shift. Repeat three times.",
    )
    parser.add_argument(
        "--b-term",
        type=parse_term,
        action="append",
        default=None,
        help="Override family B term as row_shift,col_shift. Repeat three times.",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "codes")
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ell <= 0 or args.m <= 0:
        raise SystemExit("--ell and --m must be positive")

    family = FAMILIES[args.family]
    a_terms = list(args.a_term if args.a_term is not None else family["a_terms"])
    b_terms = list(args.b_term if args.b_term is not None else family["b_terms"])
    if not a_terms or not b_terms:
        raise SystemExit("A and B must each contain at least one term")

    matrices = bivariate_bicycle_sparse(args.ell, args.m, a_terms, b_terms)
    n = 2 * args.ell * args.m
    name = args.name or f"bb_{args.family}_l{args.ell}_m{args.m}_n{n}_sparse"
    metadata = {
        "family": args.family,
        "ell": args.ell,
        "m": args.m,
        "n": n,
        "a_terms": a_terms,
        "b_terms": b_terms,
        "format": "sparse_coo",
        "construction": "bivariate_bicycle_HX=[A,B]_HZ=[B^T,A^T]",
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{name}.npz"
    np.savez_compressed(
        out_path,
        **matrices,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    print(f"wrote {out_path}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
