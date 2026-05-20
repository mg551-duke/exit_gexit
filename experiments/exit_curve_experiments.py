from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CodeData:
    path: Path
    name: str
    hx: np.ndarray
    hz: np.ndarray
    lx: np.ndarray | None
    lz: np.ndarray | None
    rank_hx: int
    rank_hz: int
    k: int

    @property
    def n(self) -> int:
        return int(self.hx.shape[1])


@dataclass(frozen=True)
class PackedRows:
    """GF(2) rows stored as Python integers for exact bit-packed elimination."""

    rows: tuple[int, ...]
    n_cols: int

    @classmethod
    def from_dense(cls, matrix: np.ndarray) -> "PackedRows":
        matrix = (matrix % 2).astype(np.uint8)
        rows = []
        for row in matrix:
            packed = 0
            for col in np.flatnonzero(row):
                packed |= 1 << int(col)
            rows.append(packed)
        return cls(rows=tuple(rows), n_cols=int(matrix.shape[1]))

    @classmethod
    def from_coo(
        cls,
        row_indices: np.ndarray,
        col_indices: np.ndarray,
        n_rows: int,
        n_cols: int,
    ) -> "PackedRows":
        """Build packed rows directly from sparse COO row/column coordinates."""
        rows = [0] * int(n_rows)
        for row, col in zip(row_indices, col_indices):
            rows[int(row)] ^= 1 << int(col)
        return cls(rows=tuple(rows), n_cols=int(n_cols))

    def rank(self, mask: int | None = None) -> int:
        if mask is None:
            return rank_int_rows(self.rows)
        return rank_int_rows(row & mask for row in self.rows)


@dataclass(frozen=True)
class ExactClassCache:
    """Prepacked matrices used by exact target-only class calculations."""

    code: CodeData
    hx: PackedRows
    hz: PackedRows
    lx: PackedRows | None
    lz: PackedRows | None
    all_mask: int
    source_format: str = "dense"


@dataclass(frozen=True)
class MatrixSupport:
    rows: list[set[int]]
    cols: list[list[int]]
    n_cols: int

    @classmethod
    def from_dense(cls, matrix: np.ndarray) -> "MatrixSupport":
        matrix = (matrix % 2).astype(np.uint8)
        rows = [set(map(int, np.flatnonzero(row))) for row in matrix]
        cols: list[list[int]] = [[] for _ in range(matrix.shape[1])]
        for row_idx, row in enumerate(rows):
            for col in row:
                cols[col].append(row_idx)
        return cls(rows=rows, cols=cols, n_cols=int(matrix.shape[1]))


class DisjointSet:
    """Small union-find helper for fast surface-code path checks."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return
        if self.rank[root_first] < self.rank[root_second]:
            root_first, root_second = root_second, root_first
        self.parent[root_second] = root_first
        if self.rank[root_first] == self.rank[root_second]:
            self.rank[root_first] += 1

    def connected(self, first: int, second: int) -> bool:
        return self.find(first) == self.find(second)


def rank_int_rows(rows: Iterable[int]) -> int:
    """Exact GF(2) rank for bit-packed rows represented as Python integers."""
    basis: dict[int, int] = {}
    rank = 0
    for row in rows:
        if add_int_row_to_basis(int(row), basis):
            rank += 1
    return rank


def add_int_row_to_basis(value: int, basis: dict[int, int]) -> bool:
    """Insert one packed GF(2) row into an existing row basis."""
    while value:
        pivot = value.bit_length() - 1
        basis_row = basis.get(pivot)
        if basis_row is None:
            basis[pivot] = value
            return True
        value ^= basis_row
    return False


def rank_increase_int_rows(
    base_rows: Iterable[int],
    extra_rows: Iterable[int],
) -> int:
    """Return how much extra_rows increase the GF(2) span of base_rows."""
    basis: dict[int, int] = {}
    rank = 0
    for row in base_rows:
        if add_int_row_to_basis(int(row), basis):
            rank += 1
    base_rank = rank
    for row in extra_rows:
        if add_int_row_to_basis(int(row), basis):
            rank += 1
    return int(rank - base_rank)


def rank_gf2_bitpacked(matrix: np.ndarray) -> int:
    """Exact GF(2) rank using bit-packed rows."""
    matrix = np.asarray(matrix)
    if matrix.size == 0:
        return 0
    if matrix.ndim != 2:
        raise ValueError("rank_gf2_bitpacked expects a 2D matrix")
    return PackedRows.from_dense(matrix).rank()


def rank_gf2_dense(matrix: np.ndarray) -> int:
    """Dense Gaussian elimination over GF(2), retained as a fallback/reference."""
    matrix = (matrix.copy() % 2).astype(np.uint8)
    if matrix.size == 0:
        return 0
    m, n = matrix.shape
    rank = 0
    for col in range(n):
        pivots = np.flatnonzero(matrix[rank:, col])
        if pivots.size == 0:
            continue
        pivot = int(pivots[0] + rank)
        if pivot != rank:
            matrix[[rank, pivot]] = matrix[[pivot, rank]]
        for row in range(m):
            if row != rank and matrix[row, col]:
                matrix[row] ^= matrix[rank]
        rank += 1
        if rank == m:
            break
    return rank


def rank_gf2(matrix: np.ndarray) -> int:
    """Default exact GF(2) rank backend."""
    return rank_gf2_bitpacked(matrix)


def load_code(path: Path) -> CodeData:
    data = np.load(path, allow_pickle=True)
    if "Hx" not in data or "Hz" not in data:
        raise ValueError(f"{path} must contain Hx and Hz arrays")
    hx = (data["Hx"] % 2).astype(np.uint8)
    hz = (data["Hz"] % 2).astype(np.uint8)
    lx = (data["Lx"] % 2).astype(np.uint8) if "Lx" in data else None
    lz = (data["Lz"] % 2).astype(np.uint8) if "Lz" in data else None
    if hx.shape[1] != hz.shape[1]:
        raise ValueError(f"{path} has mismatched Hx/Hz column counts")
    if lx is not None and lx.shape[1] != hx.shape[1]:
        raise ValueError(f"{path} has Lx column count mismatch")
    if lz is not None and lz.shape[1] != hx.shape[1]:
        raise ValueError(f"{path} has Lz column count mismatch")
    rank_hx = rank_gf2(hx)
    rank_hz = rank_gf2(hz)
    k = int(hx.shape[1] - rank_hx - rank_hz)
    return CodeData(
        path=path,
        name=path.stem,
        hx=hx,
        hz=hz,
        lx=lx,
        lz=lz,
        rank_hx=rank_hx,
        rank_hz=rank_hz,
        k=k,
    )


def npz_scalar(data: np.lib.npyio.NpzFile, key: str) -> int:
    return int(np.asarray(data[key]).item())


def has_sparse_coo_matrix(data: np.lib.npyio.NpzFile, prefix: str) -> bool:
    return (
        f"rows_{prefix}" in data
        and f"cols_{prefix}" in data
        and f"n_{prefix}" in data
        and f"m_{prefix}" in data
    )


def code_length_from_npz(path: Path) -> int:
    """Read the code length without forcing sparse HGP files into dense memory."""
    with np.load(path, allow_pickle=True) as data:
        if "Hx" in data:
            return int(data["Hx"].shape[1])
        if has_sparse_coo_matrix(data, "HX"):
            return npz_scalar(data, "m_HX")
    raise ValueError(f"{path} does not contain dense Hx or sparse HX data")


def packed_rows_from_sparse_npz(
    data: np.lib.npyio.NpzFile,
    prefix: str,
) -> PackedRows | None:
    if not has_sparse_coo_matrix(data, prefix):
        return None
    return PackedRows.from_coo(
        np.asarray(data[f"rows_{prefix}"]),
        np.asarray(data[f"cols_{prefix}"]),
        npz_scalar(data, f"n_{prefix}"),
        npz_scalar(data, f"m_{prefix}"),
    )


def build_exact_class_cache(code: CodeData) -> ExactClassCache:
    return ExactClassCache(
        code=code,
        hx=PackedRows.from_dense(code.hx),
        hz=PackedRows.from_dense(code.hz),
        lx=PackedRows.from_dense(code.lx) if code.lx is not None else None,
        lz=PackedRows.from_dense(code.lz) if code.lz is not None else None,
        all_mask=(1 << code.n) - 1,
    )


def load_sparse_exact_class_cache(path: Path) -> ExactClassCache:
    with np.load(path, allow_pickle=True) as data:
        hx = packed_rows_from_sparse_npz(data, "HX")
        hz = packed_rows_from_sparse_npz(data, "HZ")
        if hx is None or hz is None:
            raise ValueError(f"{path} must contain sparse HX and HZ COO arrays")
        if hx.n_cols != hz.n_cols:
            raise ValueError(f"{path} has mismatched sparse HX/HZ column counts")

        lx = packed_rows_from_sparse_npz(data, "LX")
        lz = packed_rows_from_sparse_npz(data, "LZ")
        if lx is not None and lx.n_cols != hx.n_cols:
            raise ValueError(f"{path} has sparse LX column count mismatch")
        if lz is not None and lz.n_cols != hx.n_cols:
            raise ValueError(f"{path} has sparse LZ column count mismatch")

    rank_hx = hx.rank()
    rank_hz = hz.rank()
    k = int(hx.n_cols - rank_hx - rank_hz)
    code = CodeData(
        path=path,
        name=path.stem,
        hx=np.empty((0, hx.n_cols), dtype=np.uint8),
        hz=np.empty((0, hx.n_cols), dtype=np.uint8),
        lx=None,
        lz=None,
        rank_hx=rank_hx,
        rank_hz=rank_hz,
        k=k,
    )
    return ExactClassCache(
        code=code,
        hx=hx,
        hz=hz,
        lx=lx,
        lz=lz,
        all_mask=(1 << code.n) - 1,
        source_format="sparse_coo",
    )


def load_exact_class_cache(path: Path) -> ExactClassCache:
    with np.load(path, allow_pickle=True) as data:
        is_dense = "Hx" in data and "Hz" in data
        is_sparse = has_sparse_coo_matrix(data, "HX") and has_sparse_coo_matrix(data, "HZ")
    if is_dense:
        return build_exact_class_cache(load_code(path))
    if is_sparse:
        return load_sparse_exact_class_cache(path)
    raise ValueError(f"{path} must contain dense Hx/Hz or sparse HX/HZ COO arrays")


def exact_class_from_logicals(
    stabilizers: PackedRows,
    logicals: PackedRows,
    kept_mask: int,
    k: int,
) -> int:
    """
    Compute logical classes supported in the erased set via logical rows.

    If K is the unerased set, logical combinations visible on K have dimension
    rank([S_K; L_K]) - rank(S_K). The remaining logical combinations can be
    represented entirely inside E, so H(C | E,S) is k minus that rank increase.
    """
    rank_increase = rank_increase_int_rows(
        (row & kept_mask for row in stabilizers.rows),
        (row & kept_mask for row in logicals.rows),
    )
    return int(k - rank_increase)


def exact_x_class_target(
    cache: ExactClassCache,
    erased_mask: int,
    *,
    use_logicals: bool = True,
) -> float:
    kept_mask = cache.all_mask ^ erased_mask
    if use_logicals and cache.lx is not None:
        return float(
            exact_class_from_logicals(
                cache.hx,
                cache.lx,
                kept_mask,
                cache.code.k,
            )
        )

    erased_size = erased_mask.bit_count()
    x_error = erased_size - cache.hz.rank(erased_mask)
    x_stabilizer = cache.code.rank_hx - cache.hx.rank(kept_mask)
    return float(x_error - x_stabilizer)


def exact_z_class_target(
    cache: ExactClassCache,
    erased_mask: int,
    *,
    use_logicals: bool = True,
) -> float:
    kept_mask = cache.all_mask ^ erased_mask
    if use_logicals and cache.lz is not None:
        return float(
            exact_class_from_logicals(
                cache.hz,
                cache.lz,
                kept_mask,
                cache.code.k,
            )
        )

    erased_size = erased_mask.bit_count()
    z_error = erased_size - cache.hx.rank(erased_mask)
    z_stabilizer = cache.code.rank_hz - cache.hz.rank(kept_mask)
    return float(z_error - z_stabilizer)


def exact_class_target_values(
    cache: ExactClassCache,
    erased_mask: int,
    *,
    component: str = "x",
    use_logicals: bool = True,
) -> dict[str, float]:
    if component == "x":
        return {
            "exact_x_class": exact_x_class_target(
                cache,
                erased_mask,
                use_logicals=use_logicals,
            )
        }
    if component == "z":
        return {
            "exact_z_class": exact_z_class_target(
                cache,
                erased_mask,
                use_logicals=use_logicals,
            )
        }
    if component == "both":
        x_class = exact_x_class_target(cache, erased_mask, use_logicals=use_logicals)
        z_class = exact_z_class_target(cache, erased_mask, use_logicals=use_logicals)
        return {
            "exact_x_class": x_class,
            "exact_z_class": z_class,
            "exact_class_total": x_class + z_class,
        }
    raise ValueError(f"unsupported exact class component: {component}")


def mask_from_bool(erased: np.ndarray) -> int:
    mask = 0
    for col in np.flatnonzero(erased):
        mask |= 1 << int(col)
    return mask


def add_threshold_edges(
    thresholds: np.ndarray,
    order: np.ndarray,
    start: int,
    stop_p: float,
    current_mask: int,
) -> tuple[int, int]:
    idx = start
    mask = current_mask
    while idx < len(order) and thresholds[int(order[idx])] < stop_p:
        mask |= 1 << int(order[idx])
        idx += 1
    return idx, mask


def exact_kernel_values(code: CodeData, erased: np.ndarray) -> dict[str, float]:
    """Exact CSS entropy dimensions from kernels and shortened stabilizers."""
    e_idx = np.flatnonzero(erased)
    k_idx = np.flatnonzero(~erased)
    e_size = int(e_idx.size)

    x_error = e_size - rank_gf2(code.hz[:, e_idx])
    z_error = e_size - rank_gf2(code.hx[:, e_idx])

    x_stabilizer = code.rank_hx - rank_gf2(code.hx[:, k_idx])
    z_stabilizer = code.rank_hz - rank_gf2(code.hz[:, k_idx])

    x_class = x_error - x_stabilizer
    z_class = z_error - z_stabilizer

    return {
        "exact_x_error": float(x_error),
        "exact_z_error": float(z_error),
        "exact_error_total": float(x_error + z_error),
        "exact_x_stabilizer": float(x_stabilizer),
        "exact_z_stabilizer": float(z_stabilizer),
        "exact_stabilizer_total": float(x_stabilizer + z_stabilizer),
        "exact_x_class": float(x_class),
        "exact_z_class": float(z_class),
        "exact_class_total": float(x_class + z_class),
        "exact_x_saved_by_stabilizers": float(x_stabilizer),
        "exact_z_saved_by_stabilizers": float(z_stabilizer),
        "exact_saved_by_stabilizers_total": float(x_stabilizer + z_stabilizer),
    }


def residual_kernel_bits(matrix: np.ndarray, residual: np.ndarray) -> int:
    residual_idx = np.flatnonzero(residual)
    return int(residual_idx.size - rank_gf2(matrix[:, residual_idx]))


def primal_peeling_remaining(support: MatrixSupport, erased: np.ndarray) -> np.ndarray:
    """Return the erasure mask remaining after standard degree-1 peeling."""
    remaining = np.asarray(erased, dtype=bool).copy()
    erased_cols = set(map(int, np.flatnonzero(remaining)))
    row_erased = [row.intersection(erased_cols) for row in support.rows]
    queue = [idx for idx, row in enumerate(row_erased) if len(row) == 1]

    head = 0
    while head < len(queue):
        row_idx = queue[head]
        head += 1
        if len(row_erased[row_idx]) != 1:
            continue
        col = next(iter(row_erased[row_idx]))
        if not remaining[col]:
            continue

        remaining[col] = False
        for other_row in support.cols[col]:
            if col not in row_erased[other_row]:
                continue
            row_erased[other_row].remove(col)
            if len(row_erased[other_row]) == 1:
                queue.append(other_row)

    return remaining


def fully_erased_pivots_from_rows(rows: list[set[int]], known: np.ndarray) -> list[int]:
    fully_erased = []
    for row in rows:
        if row and not any(known[col] for col in row):
            fully_erased.append(set(row))
    if not fully_erased:
        return []

    pivot_cols: list[int] = []
    candidate_cols = sorted(set().union(*fully_erased))
    rank_row = 0
    for col in candidate_cols:
        pivot_row = None
        for row_idx in range(rank_row, len(fully_erased)):
            if col in fully_erased[row_idx]:
                pivot_row = row_idx
                break
        if pivot_row is None:
            continue
        fully_erased[rank_row], fully_erased[pivot_row] = (
            fully_erased[pivot_row],
            fully_erased[rank_row],
        )
        for row_idx in range(len(fully_erased)):
            if row_idx != rank_row and col in fully_erased[row_idx]:
                fully_erased[row_idx] ^= fully_erased[rank_row]
        pivot_cols.append(col)
        rank_row += 1
        if rank_row == len(fully_erased):
            break
    return pivot_cols


def stabilizer_peeling_mask(
    support: MatrixSupport,
    erased: np.ndarray,
    *,
    use_rule2: bool,
) -> tuple[np.ndarray, int]:
    """
    Experiment-local dual peeling.

    This mirrors the paper rules but avoids constructing decoder messages and
    sparse matrices for every Monte Carlo sample.
    """
    remaining = np.asarray(erased, dtype=bool).copy()
    fixed = 0

    while True:
        known = ~remaining
        rows = [set(row) for row in support.rows]

        while True:
            changed = False

            for col in np.flatnonzero(known):
                incident = []
                for row_idx, row in enumerate(rows):
                    if col in row:
                        incident.append(row_idx)
                        if len(incident) > 2:
                            break
                if len(incident) == 2:
                    keep, remove = incident
                    rows[keep] ^= rows[remove]
                    del rows[remove]
                    changed = True
                    break
            if changed:
                continue

            if use_rule2:
                for row_idx, row in enumerate(rows):
                    known_cols = [col for col in row if known[col]]
                    if len(known_cols) != 1:
                        continue
                    pivot_col = known_cols[0]
                    pivot = set(row)
                    for other_idx, other_row in enumerate(rows):
                        if other_idx != row_idx and pivot_col in other_row:
                            other_row ^= pivot
                    del rows[row_idx]
                    changed = True
                    break
                if changed:
                    continue

            break

        pivots = fully_erased_pivots_from_rows(rows, known)
        new_pivots = [col for col in pivots if remaining[col]]
        if not new_pivots:
            break
        remaining[new_pivots] = False
        fixed += len(new_pivots)

    return remaining, fixed


def peeling_values(
    code: CodeData,
    supports: dict[str, MatrixSupport],
    erased: np.ndarray,
    *,
    use_rule2: bool,
) -> dict[str, float]:
    """Algorithmic entropy proxies from primal and stabilizer peeling."""
    x_plain_mask = primal_peeling_remaining(supports["hz"], erased)
    z_plain_mask = primal_peeling_remaining(supports["hx"], erased)
    x_plain_residual = int(x_plain_mask.sum())
    z_plain_residual = int(z_plain_mask.sum())
    x_unassisted_guess = residual_kernel_bits(code.hz, x_plain_mask)
    z_unassisted_guess = residual_kernel_bits(code.hx, z_plain_mask)

    x_after_stab_mask, x_stab_found = stabilizer_peeling_mask(
        supports["hx"], erased, use_rule2=use_rule2
    )
    z_after_stab_mask, z_stab_found = stabilizer_peeling_mask(
        supports["hz"], erased, use_rule2=use_rule2
    )

    x_after_dual_primal_mask = primal_peeling_remaining(supports["hz"], x_after_stab_mask)
    z_after_dual_primal_mask = primal_peeling_remaining(supports["hx"], z_after_stab_mask)
    x_after_dual_primal = int(x_after_dual_primal_mask.sum())
    z_after_dual_primal = int(z_after_dual_primal_mask.sum())
    x_aided_guess = residual_kernel_bits(code.hz, x_after_dual_primal_mask)
    z_aided_guess = residual_kernel_bits(code.hx, z_after_dual_primal_mask)

    x_saved_guess = max(0, x_unassisted_guess - x_aided_guess)
    z_saved_guess = max(0, z_unassisted_guess - z_aided_guess)

    return {
        "peel_x_residual_qubits": float(x_plain_residual),
        "peel_z_residual_qubits": float(z_plain_residual),
        "peel_residual_qubits_total": float(x_plain_residual + z_plain_residual),
        "peel_x_unassisted_guess": float(x_unassisted_guess),
        "peel_z_unassisted_guess": float(z_unassisted_guess),
        "peel_unassisted_guess_total": float(x_unassisted_guess + z_unassisted_guess),
        "peel_x_stabilizer_found": float(x_stab_found),
        "peel_z_stabilizer_found": float(z_stab_found),
        "peel_stabilizer_found_total": float(x_stab_found + z_stab_found),
        "peel_x_saved_guess": float(x_saved_guess),
        "peel_z_saved_guess": float(z_saved_guess),
        "peel_saved_guess_total": float(x_saved_guess + z_saved_guess),
        "peel_x_aided_guess": float(x_aided_guess),
        "peel_z_aided_guess": float(z_aided_guess),
        "peel_aided_guess_total": float(x_aided_guess + z_aided_guess),
        "dual_primal_x_residual": float(x_after_dual_primal),
        "dual_primal_z_residual": float(z_after_dual_primal),
        "dual_primal_residual_total": float(x_after_dual_primal + z_after_dual_primal),
    }


def summarize(samples: list[dict[str, float]], n: int, k: int) -> dict[str, float]:
    keys = sorted(samples[0])
    out: dict[str, float] = {}
    for key in keys:
        values = np.array([sample[key] for sample in samples], dtype=float)
        mean = float(values.mean())
        stderr = float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
        out[key] = mean
        out[f"{key}_stderr"] = stderr
        out[f"{key}_norm"] = mean / (2 * n)
        out[f"{key}_norm_stderr"] = stderr / (2 * n)
        if "_x_" in key or "_z_" in key:
            out[f"{key}_component_norm"] = mean / n
            out[f"{key}_component_norm_stderr"] = stderr / n
        is_class_quantity = "class" in key or "aided_guess" in key
        if k > 0 and is_class_quantity:
            denom = k if "_x_" in key or "_z_" in key else 2 * k
            out[f"{key}_logical_norm"] = mean / denom
            out[f"{key}_logical_norm_stderr"] = stderr / denom
    return out


def summarize_derivative_samples(
    samples: list[dict[str, float]],
    n: int,
    k: int,
) -> dict[str, float]:
    """Summarize per-sample d/dp estimates using the same naming as curve gradients."""
    keys = sorted(samples[0])
    out: dict[str, float] = {}
    for key in keys:
        values = np.array([sample[key] for sample in samples], dtype=float)
        mean = float(values.mean())
        stderr = float(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
        out[f"{key}_dp"] = mean
        out[f"{key}_dp_stderr"] = stderr
        out[f"{key}_norm_dp"] = mean / (2 * n)
        out[f"{key}_norm_dp_stderr"] = stderr / (2 * n)
        if "_x_" in key or "_z_" in key:
            out[f"{key}_component_norm_dp"] = mean / n
            out[f"{key}_component_norm_dp_stderr"] = stderr / n
        is_class_quantity = "class" in key or "aided_guess" in key
        if k > 0 and is_class_quantity:
            denom = k if "_x_" in key or "_z_" in key else 2 * k
            out[f"{key}_logical_norm_dp"] = mean / denom
            out[f"{key}_logical_norm_dp_stderr"] = stderr / denom
    return out


def add_finite_difference_derivatives(points: list[dict[str, float]]) -> None:
    """Add d/dp columns for every entropy/proxy curve."""
    if len(points) < 2:
        return
    ps = np.array([point["p"] for point in points], dtype=float)
    curve_keys = sorted(
        key
        for key in points[0]
        if key not in {"p", "runs"}
        and not key.endswith("_stderr")
        and not key.endswith("_dp")
    )
    for key in curve_keys:
        values = np.array([point[key] for point in points], dtype=float)
        deriv = np.gradient(values, ps)
        for point, value in zip(points, deriv):
            point[f"{key}_dp"] = float(value)


def curve_areas(points: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Calculate area under finite-difference derivatives for each curve."""
    if len(points) < 2:
        return {}
    ps = np.array([point["p"] for point in points], dtype=float)
    out: dict[str, dict[str, float]] = {}
    curve_keys = sorted(
        key
        for key in points[0]
        if key not in {"p", "runs"}
        and not key.endswith("_stderr")
        and not key.endswith("_dp")
    )
    for key in curve_keys:
        values = np.array([point[key] for point in points], dtype=float)
        deriv_key = f"{key}_dp"
        if deriv_key not in points[0]:
            continue
        deriv = np.array([point[deriv_key] for point in points], dtype=float)
        out[key] = {
            "endpoint_delta": float(values[-1] - values[0]),
            "trapezoid_derivative_area": float(np.trapezoid(deriv, ps)),
            "value_at_p0": float(values[0]),
            "value_at_p1": float(values[-1]),
        }
    return out


def derivative_areas(points: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Calculate areas under derivative-only curves."""
    if len(points) < 2:
        return {}
    ps = np.array([point["p"] for point in points], dtype=float)
    out: dict[str, dict[str, float]] = {}
    curve_keys = sorted(
        key
        for key in points[0]
        if key.endswith("_dp") and not key.endswith("_stderr")
    )
    for key in curve_keys:
        values = np.array([point[key] for point in points], dtype=float)
        out[key] = {
            "trapezoid_derivative_area": float(np.trapezoid(values, ps)),
            "value_min": float(values.min()),
            "value_max": float(values.max()),
        }
    return out


def quantity_notation() -> dict[str, dict[str, str]]:
    return {
        "exact_x_error": {
            "notation": "H(x_E | E, S)",
            "description": "Raw X-error posterior entropy in bits; equals |E| - rank(H_Z,E).",
        },
        "exact_z_error": {
            "notation": "H(z_E | E, s_X)",
            "description": "Raw Z-error posterior entropy in bits; equals |E| - rank(H_X,E).",
        },
        "exact_error_total": {
            "notation": "H(x_E, z_E | E, S)",
            "description": "Raw Pauli posterior entropy in bits.",
        },
        "exact_x_stabilizer": {
            "notation": "dim S_X(E)",
            "description": "Dimension of X-type stabilizers supported inside E.",
        },
        "exact_z_stabilizer": {
            "notation": "dim S_Z(E)",
            "description": "Dimension of Z-type stabilizers supported inside E.",
        },
        "exact_stabilizer_total": {
            "notation": "dim S_X(E) + dim S_Z(E)",
            "description": "Total erased-stabilizer degeneracy in bits.",
        },
        "exact_x_class": {
            "notation": "H(C_X | E, S)",
            "description": "X-correction-class entropy in bits; equals H(x_E | E,S) - dim S_X(E).",
        },
        "exact_z_class": {
            "notation": "H(C_Z | E, s_X)",
            "description": "Z-correction-class entropy in bits; equals H(z_E | E,s_X) - dim S_Z(E).",
        },
        "exact_class_total": {
            "notation": "H(C | E, S)",
            "description": "Quantum correction-class entropy in bits.",
        },
        "exact_x_saved_by_stabilizers": {
            "notation": "H(x_E | E,S) - H(C_X | E,S)",
            "description": "Exact X-side guess reduction from erased stabilizer degeneracy.",
        },
        "exact_z_saved_by_stabilizers": {
            "notation": "H(z_E | E,s_X) - H(C_Z | E,s_X)",
            "description": "Exact Z-side guess reduction from erased stabilizer degeneracy.",
        },
        "exact_saved_by_stabilizers_total": {
            "notation": "H(x_E,z_E | E,S) - H(C | E,S)",
            "description": "Exact total guess reduction from erased stabilizer degeneracy.",
        },
        "peel_x_residual_qubits": {
            "notation": "|R_X^peel(E)|",
            "description": "Number of X-side erased qubits left unresolved by primal peeling.",
        },
        "peel_z_residual_qubits": {
            "notation": "|R_Z^peel(E)|",
            "description": "Number of Z-side erased qubits left unresolved by primal peeling.",
        },
        "peel_residual_qubits_total": {
            "notation": "|R_X^peel(E)| + |R_Z^peel(E)|",
            "description": "Total unresolved erased qubits left by primal peeling.",
        },
        "peel_x_unassisted_guess": {
            "notation": "H_peel(x_E | E, S)",
            "description": "Guess bits required after primal peeling without stabilizer aid; approximates H(x_E | E,S).",
        },
        "peel_z_unassisted_guess": {
            "notation": "H_peel(z_E | E, s_X)",
            "description": "Guess bits required after primal peeling without stabilizer aid; approximates H(z_E | E,s_X).",
        },
        "peel_unassisted_guess_total": {
            "notation": "H_peel(x_E,z_E | E, S)",
            "description": "Total unassisted peeling guess bits; approximates H(x_E,z_E | E,S).",
        },
        "peel_x_stabilizer_found": {
            "notation": "dim S_X^peel(E)",
            "description": "X-type erased stabilizers found by dual peeling.",
        },
        "peel_z_stabilizer_found": {
            "notation": "dim S_Z^peel(E)",
            "description": "Z-type erased stabilizers found by dual peeling.",
        },
        "peel_stabilizer_found_total": {
            "notation": "dim S_X^peel(E) + dim S_Z^peel(E)",
            "description": "Total erased stabilizers found by dual peeling.",
        },
        "peel_x_saved_guess": {
            "notation": "H_peel(x_E | E,S) - H_peel(C_X | E,S)",
            "description": "Algorithmic X-side guess reduction from stabilizer aid.",
        },
        "peel_z_saved_guess": {
            "notation": "H_peel(z_E | E,s_X) - H_peel(C_Z | E,s_X)",
            "description": "Algorithmic Z-side guess reduction from stabilizer aid.",
        },
        "peel_saved_guess_total": {
            "notation": "H_peel(x_E,z_E | E,S) - H_peel(C | E,S)",
            "description": "Algorithmic total guess reduction from stabilizer aid.",
        },
        "peel_x_aided_guess": {
            "notation": "H_peel(C_X | E, S)",
            "description": "Guess bits required after stabilizer peeling and primal peeling; approximates H(C_X | E,S).",
        },
        "peel_z_aided_guess": {
            "notation": "H_peel(C_Z | E, s_X)",
            "description": "Guess bits required after stabilizer peeling and primal peeling; approximates H(C_Z | E,s_X).",
        },
        "peel_aided_guess_total": {
            "notation": "H_peel(C | E, S)",
            "description": "Total guess bits required after stabilizer peeling and primal peeling; approximates H(C | E,S).",
        },
        "dual_primal_residual_total": {
            "notation": "|R_X^dual+primal(E)| + |R_Z^dual+primal(E)|",
            "description": "Residual qubit count after stabilizer peeling followed by primal peeling.",
        },
    }


def run_code(
    code_path: Path,
    ps: Iterable[float],
    runs: int,
    *,
    seed: int = 0,
    include_exact: bool = True,
    include_peeling: bool = True,
    use_rule2: bool = False,
) -> dict:
    code = load_code(code_path)
    rng = np.random.default_rng(seed)
    supports = {
        "hx": MatrixSupport.from_dense(code.hx),
        "hz": MatrixSupport.from_dense(code.hz),
    }
    points = []
    t0 = time.perf_counter()
    for p in ps:
        samples = []
        for _ in range(runs):
            erased = rng.random(code.n) < p
            sample: dict[str, float] = {}
            if include_exact:
                sample.update(exact_kernel_values(code, erased))
            if include_peeling:
                sample.update(
                    peeling_values(
                        code,
                        supports,
                        erased,
                        use_rule2=use_rule2,
                    )
                )
            samples.append(sample)
        point = {"p": float(p), "runs": int(runs)}
        point.update(summarize(samples, code.n, code.k))
        points.append(point)

    add_finite_difference_derivatives(points)
    areas = curve_areas(points)

    return {
        "code": {
            "path": str(code_path),
            "name": code.name,
            "n": code.n,
            "rank_hx": code.rank_hx,
            "rank_hz": code.rank_hz,
            "k": code.k,
            "rate": code.k / code.n,
        },
        "config": {
            "runs": runs,
            "seed": seed,
            "include_exact": include_exact,
            "include_peeling": include_peeling,
            "dual_rule2": use_rule2,
            "units": {
                "unsuffixed_columns": "bits",
                "_norm": "bits divided by 2n",
                "_component_norm": "single CSS-component bits divided by n",
                "_logical_norm": "X/Z class bits divided by k; total class bits divided by 2k",
                "_dp": "finite-difference derivative with respect to erasure probability",
            },
        },
        "quantity_notation": quantity_notation(),
        "areas": areas,
        "elapsed_seconds": time.perf_counter() - t0,
        "points": points,
    }


def run_exact_class_target_code(
    code_path: Path,
    ps: Iterable[float],
    runs: int,
    *,
    seed: int = 0,
    component: str = "x",
    use_logicals: bool = True,
    use_rule2: bool = False,
) -> dict:
    """Average only exact correction-class entropy using packed rank queries."""
    cache = load_exact_class_cache(code_path)
    code = cache.code
    rng = np.random.default_rng(seed)
    points = []
    t0 = time.perf_counter()

    for p in ps:
        samples = []
        for _ in range(runs):
            erased = rng.random(code.n) < p
            erased_mask = mask_from_bool(erased)
            samples.append(
                exact_class_target_values(
                    cache,
                    erased_mask,
                    component=component,
                    use_logicals=use_logicals,
                )
            )
        point = {"p": float(p), "runs": int(runs)}
        point.update(summarize(samples, code.n, code.k))
        points.append(point)

    add_finite_difference_derivatives(points)
    areas = curve_areas(points)

    return {
        "code": {
            "path": str(code_path),
            "name": code.name,
            "n": code.n,
            "rank_hx": code.rank_hx,
            "rank_hz": code.rank_hz,
            "k": code.k,
            "rate": code.k / code.n,
        },
        "config": {
            "runs": runs,
            "seed": seed,
            "include_exact": True,
            "include_peeling": False,
            "dual_rule2": use_rule2,
            "exact_class_target_only": True,
            "exact_class_component": component,
            "logical_rank_class": bool(
                use_logicals
                and (
                    (component == "x" and cache.lx is not None)
                    or (component == "z" and cache.lz is not None)
                    or (
                        component == "both"
                        and cache.lx is not None
                        and cache.lz is not None
                    )
                )
            ),
            "rank_backend": "bitpacked",
            "source_format": cache.source_format,
            "units": {
                "unsuffixed_columns": "bits",
                "_norm": "bits divided by 2n",
                "_component_norm": "single CSS-component bits divided by n",
                "_logical_norm": "X/Z class bits divided by k; total class bits divided by 2k",
                "_dp": "finite-difference derivative with respect to erasure probability",
            },
        },
        "quantity_notation": quantity_notation(),
        "areas": areas,
        "elapsed_seconds": time.perf_counter() - t0,
        "points": points,
    }


def run_paired_derivative_code(
    code_path: Path,
    ps: Iterable[float],
    runs: int,
    *,
    seed: int = 0,
    include_exact: bool = True,
    include_peeling: bool = True,
    use_rule2: bool = False,
) -> dict:
    """
    Estimate derivative curves by differencing nested erasure samples first.

    Each Monte Carlo sample draws one threshold vector u in [0, 1)^n and reuses
    it for every p via E(p) = {i : u_i < p}. Per-sample finite differences are
    then averaged, reducing derivative variance relative to differencing
    independently sampled mean curves.
    """
    code = load_code(code_path)
    p_values = [float(p) for p in ps]
    if len(p_values) < 2:
        raise ValueError("paired derivative estimates require at least two p values")
    if any(right <= left for left, right in zip(p_values, p_values[1:])):
        raise ValueError("p values must be strictly increasing")

    rng = np.random.default_rng(seed)
    supports = {
        "hx": MatrixSupport.from_dense(code.hx),
        "hz": MatrixSupport.from_dense(code.hz),
    }
    derivative_samples: list[list[dict[str, float]]] = [[] for _ in p_values]
    t0 = time.perf_counter()

    for _ in range(runs):
        thresholds = rng.random(code.n)
        sample_values = []
        for p in p_values:
            erased = thresholds < p
            sample: dict[str, float] = {}
            if include_exact:
                sample.update(exact_kernel_values(code, erased))
            if include_peeling:
                sample.update(
                    peeling_values(
                        code,
                        supports,
                        erased,
                        use_rule2=use_rule2,
                    )
                )
            sample_values.append(sample)

        for idx in range(len(p_values)):
            if idx == 0:
                low_idx, high_idx = 0, 1
            elif idx == len(p_values) - 1:
                low_idx, high_idx = idx - 1, idx
            else:
                low_idx, high_idx = idx - 1, idx + 1

            denom = p_values[high_idx] - p_values[low_idx]
            low = sample_values[low_idx]
            high = sample_values[high_idx]
            derivative_samples[idx].append(
                {key: (high[key] - low[key]) / denom for key in low}
            )

    points = []
    for idx, p in enumerate(p_values):
        if idx == 0:
            low_idx, high_idx = 0, 1
        elif idx == len(p_values) - 1:
            low_idx, high_idx = idx - 1, idx
        else:
            low_idx, high_idx = idx - 1, idx + 1
        point = {
            "p": p,
            "p_low": p_values[low_idx],
            "p_high": p_values[high_idx],
            "runs": int(runs),
        }
        point.update(summarize_derivative_samples(derivative_samples[idx], code.n, code.k))
        points.append(point)

    return {
        "code": {
            "path": str(code_path),
            "name": code.name,
            "n": code.n,
            "rank_hx": code.rank_hx,
            "rank_hz": code.rank_hz,
            "k": code.k,
            "rate": code.k / code.n,
        },
        "config": {
            "runs": runs,
            "seed": seed,
            "include_exact": include_exact,
            "include_peeling": include_peeling,
            "dual_rule2": use_rule2,
            "paired_derivative": True,
            "units": {
                "_dp": "paired finite-difference derivative with respect to erasure probability",
                "_norm_dp": "paired derivative of bits divided by 2n",
                "_component_norm_dp": "paired derivative of single CSS-component bits divided by n",
                "_logical_norm_dp": "paired derivative of X/Z class bits divided by k; total class bits divided by 2k",
            },
        },
        "quantity_notation": quantity_notation(),
        "areas": derivative_areas(points),
        "elapsed_seconds": time.perf_counter() - t0,
        "points": points,
    }


def run_paired_exact_class_target_code(
    code_path: Path,
    ps: Iterable[float],
    runs: int,
    *,
    seed: int = 0,
    component: str = "x",
    use_logicals: bool = True,
    use_rule2: bool = False,
) -> dict:
    """
    Paired derivative for exact class entropy only.

    This is the general-code fast path: each sample draws one threshold vector,
    updates the erased mask incrementally as p increases, and evaluates only
    the requested class entropy using bit-packed GF(2) rank. If Lx/Lz are
    present, the logical-rank formula is used; otherwise it falls back to the
    exact kernel/stabilizer rank identity without requiring precomputed sparse
    files.
    """
    cache = load_exact_class_cache(code_path)
    code = cache.code
    p_values = [float(p) for p in ps]
    if len(p_values) < 2:
        raise ValueError("paired derivative estimates require at least two p values")
    if any(right <= left for left, right in zip(p_values, p_values[1:])):
        raise ValueError("p values must be strictly increasing")

    rng = np.random.default_rng(seed)
    value_samples: list[list[dict[str, float]]] = [[] for _ in p_values]
    derivative_samples: list[list[dict[str, float]]] = [[] for _ in p_values]
    t0 = time.perf_counter()

    for _ in range(runs):
        thresholds = rng.random(code.n)
        order = np.argsort(thresholds, kind="mergesort")
        next_col = 0
        erased_mask = 0
        sample_values = []

        for p in p_values:
            next_col, erased_mask = add_threshold_edges(
                thresholds,
                order,
                next_col,
                p,
                erased_mask,
            )
            sample_values.append(
                exact_class_target_values(
                    cache,
                    erased_mask,
                    component=component,
                    use_logicals=use_logicals,
                )
            )
            value_samples[len(sample_values) - 1].append(sample_values[-1])

        for idx in range(len(p_values)):
            if idx == 0:
                low_idx, high_idx = 0, 1
            elif idx == len(p_values) - 1:
                low_idx, high_idx = idx - 1, idx
            else:
                low_idx, high_idx = idx - 1, idx + 1

            denom = p_values[high_idx] - p_values[low_idx]
            low = sample_values[low_idx]
            high = sample_values[high_idx]
            derivative_samples[idx].append(
                {key: (high[key] - low[key]) / denom for key in low}
            )

    points = []
    for idx, p in enumerate(p_values):
        if idx == 0:
            low_idx, high_idx = 0, 1
        elif idx == len(p_values) - 1:
            low_idx, high_idx = idx - 1, idx
        else:
            low_idx, high_idx = idx - 1, idx + 1
        point = {
            "p": p,
            "p_low": p_values[low_idx],
            "p_high": p_values[high_idx],
            "runs": int(runs),
        }
        point.update(summarize(value_samples[idx], code.n, code.k))
        point.update(summarize_derivative_samples(derivative_samples[idx], code.n, code.k))
        points.append(point)

    return {
        "code": {
            "path": str(code_path),
            "name": code.name,
            "n": code.n,
            "rank_hx": code.rank_hx,
            "rank_hz": code.rank_hz,
            "k": code.k,
            "rate": code.k / code.n,
        },
        "config": {
            "runs": runs,
            "seed": seed,
            "include_exact": True,
            "include_peeling": False,
            "dual_rule2": use_rule2,
            "paired_derivative": True,
            "incremental_nested_erasure": True,
            "exact_class_target_only": True,
            "exact_class_component": component,
            "logical_rank_class": bool(
                use_logicals
                and (
                    (component == "x" and cache.lx is not None)
                    or (component == "z" and cache.lz is not None)
                    or (
                        component == "both"
                        and cache.lx is not None
                        and cache.lz is not None
                    )
                )
            ),
            "rank_backend": "bitpacked",
            "source_format": cache.source_format,
            "units": {
                "_dp": "paired finite-difference derivative with respect to erasure probability",
                "_norm_dp": "paired derivative of bits divided by 2n",
                "_component_norm_dp": "paired derivative of single CSS-component bits divided by n",
                "_logical_norm_dp": "paired derivative of X/Z class bits divided by k; total class bits divided by 2k",
            },
        },
        "quantity_notation": quantity_notation(),
        "areas": derivative_areas(points),
        "elapsed_seconds": time.perf_counter() - t0,
        "points": points,
    }


def surface_distance_from_code_path(code_path: Path) -> int:
    match = re.search(r"surface(\d+)_HxHzLxLz$", code_path.stem)
    if match is None:
        raise ValueError(f"could not infer surface distance from {code_path.name}")
    return int(match.group(1))


def surface_x_edges(d: int) -> list[tuple[int, int]]:
    """
    Return graph edges for the repo's standard planar surface-code layout.

    The first d^2 qubits are vertical edges in a (d+1)-by-d vertex grid.
    The remaining (d-1)^2 qubits are horizontal interior edges. An X logical
    class is supported in E exactly when erased edges connect the top and
    bottom boundaries of this graph.
    """
    edges: list[tuple[int, int]] = []

    def vertex(row: int, col: int) -> int:
        return row * d + col

    for row in range(d):
        for col in range(d):
            edges.append((vertex(row, col), vertex(row + 1, col)))

    for row in range(d - 1):
        for col in range(d - 1):
            edges.append((vertex(row + 1, col), vertex(row + 1, col + 1)))

    return edges


def surface_x_class_values_from_thresholds(
    thresholds: np.ndarray,
    p_values: list[float],
    d: int,
    edges: list[tuple[int, int]],
) -> list[float]:
    """Return H(C_X|E,S) in bits, 0 or 1, for every p in one nested sample."""
    vertex_count = (d + 1) * d
    top = vertex_count
    bottom = vertex_count + 1
    dsu = DisjointSet(vertex_count + 2)

    for col in range(d):
        dsu.union(top, col)
        dsu.union(bottom, d * d + col)

    bins = np.searchsorted(p_values, thresholds, side="right")
    values: list[float] = []
    for idx, _p in enumerate(p_values):
        for edge_idx in np.flatnonzero(bins == idx):
            dsu.union(*edges[edge_idx])
        connected = dsu.connected(top, bottom)
        values.append(float(connected))
        if connected:
            values.extend([1.0] * (len(p_values) - idx - 1))
            break
    return values


def run_surface_paired_derivative_code(
    code_path: Path,
    ps: Iterable[float],
    runs: int,
    *,
    seed: int = 0,
    use_rule2: bool = False,
) -> dict:
    """
    Fast paired derivative for repo-layout surface codes.

    This computes only the exact one-sided class curve H(C_X|E,S). For the
    one-logical-qubit planar surface codes generated in this repo, this value
    is the indicator that erased qubit-edges contain a top-to-bottom X logical
    path. That avoids thousands of dense GF(2) rank computations.
    """
    d = surface_distance_from_code_path(code_path)
    n = d * d + (d - 1) * (d - 1)
    p_values = [float(p) for p in ps]
    if len(p_values) < 2:
        raise ValueError("paired derivative estimates require at least two p values")
    if any(right <= left for left, right in zip(p_values, p_values[1:])):
        raise ValueError("p values must be strictly increasing")

    rng = np.random.default_rng(seed)
    edges = surface_x_edges(d)
    value_samples: list[list[dict[str, float]]] = [[] for _ in p_values]
    derivative_samples: list[list[dict[str, float]]] = [[] for _ in p_values]
    t0 = time.perf_counter()

    for _ in range(runs):
        thresholds = rng.random(n)
        sample_values = surface_x_class_values_from_thresholds(
            thresholds,
            p_values,
            d,
            edges,
        )
        for idx, value in enumerate(sample_values):
            value_samples[idx].append({"exact_x_class": value})

        for idx in range(len(p_values)):
            if idx == 0:
                low_idx, high_idx = 0, 1
            elif idx == len(p_values) - 1:
                low_idx, high_idx = idx - 1, idx
            else:
                low_idx, high_idx = idx - 1, idx + 1
            denom = p_values[high_idx] - p_values[low_idx]
            derivative_samples[idx].append(
                {
                    "exact_x_class": (
                        sample_values[high_idx] - sample_values[low_idx]
                    )
                    / denom
                }
            )

    points = []
    for idx, p in enumerate(p_values):
        if idx == 0:
            low_idx, high_idx = 0, 1
        elif idx == len(p_values) - 1:
            low_idx, high_idx = idx - 1, idx
        else:
            low_idx, high_idx = idx - 1, idx + 1
        point = {
            "p": p,
            "p_low": p_values[low_idx],
            "p_high": p_values[high_idx],
            "runs": int(runs),
        }
        point.update(summarize(value_samples[idx], n, 1))
        point.update(summarize_derivative_samples(derivative_samples[idx], n, 1))
        points.append(point)

    rank = d * (d - 1)
    return {
        "code": {
            "path": str(code_path),
            "name": code_path.stem,
            "n": n,
            "rank_hx": rank,
            "rank_hz": rank,
            "k": 1,
            "rate": 1 / n,
            "surface_distance": d,
        },
        "config": {
            "runs": runs,
            "seed": seed,
            "include_exact": True,
            "include_peeling": False,
            "dual_rule2": use_rule2,
            "paired_derivative": True,
            "surface_fast_x_class": True,
            "units": {
                "unsuffixed_columns": "bits",
                "_norm": "bits divided by 2n",
                "_component_norm": "single CSS-component bits divided by n",
                "_logical_norm": "X class bits divided by k",
                "_dp": "paired finite-difference derivative with respect to erasure probability",
                "_norm_dp": "paired derivative of bits divided by 2n",
                "_component_norm_dp": "paired derivative of single CSS-component bits divided by n",
                "_logical_norm_dp": "paired derivative of X class bits divided by k",
            },
        },
        "quantity_notation": quantity_notation(),
        "areas": derivative_areas(points),
        "elapsed_seconds": time.perf_counter() - t0,
        "points": points,
    }


def representative_runs(n: int) -> int:
    if n <= 50:
        return 5000
    if n <= 100:
        return 3000
    if n <= 250:
        return 1500
    if n <= 350:
        return 1000
    if n <= 700:
        return 400
    return 200


def default_ps(step: float) -> list[float]:
    count = int(round(1.0 / step))
    return [round(i * step, 10) for i in range(count + 1)]


def centered_ps(
    *,
    edge_step: float = 0.1,
    shoulder_step: float = 0.05,
    center_step: float = 0.02,
) -> list[float]:
    """Return a nonuniform p grid with original-like center spacing and sparse edges."""
    if edge_step <= 0 or shoulder_step <= 0 or center_step <= 0:
        raise ValueError("p grid steps must be positive")

    ranges = [
        (0.0, 0.2, edge_step),
        (0.2, 0.35, shoulder_step),
        (0.35, 0.65, center_step),
        (0.65, 0.8, shoulder_step),
        (0.8, 1.0, edge_step),
    ]
    values: set[float] = set()
    for start, stop, step in ranges:
        count = int(round((stop - start) / step))
        for idx in range(count + 1):
            value = start + idx * step
            if start - 1e-12 <= value <= stop + 1e-12:
                values.add(round(value, 10))
    values.update({0.0, 0.5, 1.0})
    return sorted(values)


def write_outputs(result: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = result["code"]["name"]
    suffix = "rule2" if result["config"]["dual_rule2"] else "rule1"
    json_path = out_dir / f"{name}_exit_{suffix}.json"
    csv_path = out_dir / f"{name}_exit_{suffix}.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    fieldnames = sorted({key for point in result["points"] for key in point})
    if "p" in fieldnames:
        fieldnames.remove("p")
        fieldnames.insert(0, "p")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result["points"])

    return json_path, csv_path


def write_paired_derivative_outputs(result: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = result["code"]["name"]
    suffix = "rule2" if result["config"]["dual_rule2"] else "rule1"
    json_path = out_dir / f"{name}_exit_{suffix}_paired_derivative.json"
    csv_path = out_dir / f"{name}_exit_{suffix}_paired_derivative.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    fieldnames = sorted({key for point in result["points"] for key in point})
    for leading in reversed(["p", "p_low", "p_high"]):
        if leading in fieldnames:
            fieldnames.remove(leading)
            fieldnames.insert(0, leading)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result["points"])

    return json_path, csv_path


def plot_result(result: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    points = result["points"]
    p = [pt["p"] for pt in points]
    name = result["code"]["name"]
    suffix = "rule2" if result["config"]["dual_rule2"] else "rule1"

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    bit_series = [
        ("exact_x_error", "H(x_E | E,S) exact"),
        ("exact_x_class", "H(C_X | E,S) exact"),
        ("exact_x_saved_by_stabilizers", "exact saved guesses"),
        ("peel_x_unassisted_guess", "H_peel(x_E | E,S)"),
        ("peel_x_aided_guess", "H_peel(C_X | E,S)"),
        ("peel_x_saved_guess", "peeling saved guesses"),
    ]
    for key, label in bit_series:
        if key in points[0]:
            axes[0].plot(p, [pt[key] for pt in points], marker=".", linewidth=1.4, label=label)
    axes[0].axhline(result["code"]["k"], color="black", linestyle="--", linewidth=1, label="k bits")
    axes[0].set_xlabel("erasure probability")
    axes[0].set_ylabel("entropy or residual proxy (bits)")
    axes[0].set_title(f"{name}: bit scale")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    area_series = [
        ("exact_x_class_component_norm_dp", "d[H(C_X|E,S)/n]/dp"),
        ("peel_x_aided_guess_component_norm_dp", "d[H_peel(C_X|E,S)/n]/dp"),
        ("peel_x_unassisted_guess_component_norm_dp", "d[H_peel(x_E|E,S)/n]/dp"),
    ]
    for key, label in area_series:
        if key in points[0]:
            axes[1].plot(p, [pt[key] for pt in points], marker=".", linewidth=1.4, label=label)
    axes[1].set_xlabel("erasure probability")
    axes[1].set_ylabel("EXIT estimate d(entropy / 2n)/dp")
    axes[1].set_title(f"{name}: derivative scale")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)

    logical_series = [
        ("exact_x_class_logical_norm", "H(C_X|E,S) / k"),
        ("peel_x_aided_guess_logical_norm", "H_peel(C_X|E,S) / k"),
    ]
    for key, label in logical_series:
        if key in points[0]:
            axes[2].plot(p, [pt[key] for pt in points], marker=".", linewidth=1.4, label=label)
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1, label="full logical entropy")
    axes[2].set_xlabel("erasure probability")
    axes[2].set_ylabel("logical-normalized class entropy")
    axes[2].set_title(f"{name}: logical scale")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(fontsize=8)
    fig.tight_layout()

    path = out_dir / f"{name}_exit_{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_paired_derivative_result(result: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    points = result["points"]
    p = np.array([pt["p"] for pt in points], dtype=float)
    name = result["code"]["name"]
    suffix = "rule2" if result["config"]["dual_rule2"] else "rule1"
    n = float(result["code"]["n"])
    areas = result.get("areas", {})

    logical_series = [
        ("exact_x_class_logical_norm", "H(C_X|E,S) / k"),
        ("peel_x_aided_guess_logical_norm", "H_peel(C_X|E,S) / k"),
    ]
    has_logical_panel = any(key in points[0] for key, _ in logical_series)
    if has_logical_panel:
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
        ax = axes[0]
        logical_ax = axes[1]
    else:
        fig, ax = plt.subplots(figsize=(8.4, 5.0))
        logical_ax = None

    area_series = [
        ("exact_x_class_component_norm_dp", "d[H(C_X|E,S)/n]/dp"),
        ("peel_x_aided_guess_component_norm_dp", "d[H_peel(C_X|E,S)/n]/dp"),
        ("peel_x_unassisted_guess_component_norm_dp", "d[H_peel(x_E|E,S)/n]/dp"),
    ]
    area_lines = []
    for key, label in area_series:
        if key not in points[0]:
            continue
        values = np.array([pt[key] for pt in points], dtype=float)
        line = ax.plot(p, values, marker=".", linewidth=1.5, label=label)[0]
        area = areas.get(key, {}).get("trapezoid_derivative_area")
        if area is None:
            area = float(np.trapezoid(values, p))
        area_lines.append(f"{label}: area={area:.5g} ~= {area * n:.2f}/n")
        stderr_key = f"{key}_stderr"
        if stderr_key in points[0]:
            stderr = np.array([pt[stderr_key] for pt in points], dtype=float)
            ax.fill_between(
                p,
                values - stderr,
                values + stderr,
                color=line.get_color(),
                alpha=0.12,
                linewidth=0,
            )
    ax.set_xlabel("erasure probability")
    ax.set_ylabel("paired EXIT estimate d(entropy / n)/dp")
    ax.set_title(f"{name}: paired derivative scale ({len(points)} p-points, {result['config']['runs']} runs)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    if area_lines:
        ax.text(
            0.02,
            0.98,
            "\n".join(area_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "0.75",
                "alpha": 0.9,
            },
        )

    if logical_ax is not None:
        for key, label in logical_series:
            if key not in points[0]:
                continue
            values = np.array([pt[key] for pt in points], dtype=float)
            line = logical_ax.plot(p, values, marker=".", linewidth=1.5, label=label)[0]
            stderr_key = f"{key}_stderr"
            if stderr_key in points[0]:
                stderr = np.array([pt[stderr_key] for pt in points], dtype=float)
                logical_ax.fill_between(
                    p,
                    values - stderr,
                    values + stderr,
                    color=line.get_color(),
                    alpha=0.12,
                    linewidth=0,
                )
        logical_ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="full logical entropy")
        logical_ax.set_xlabel("erasure probability")
        logical_ax.set_ylabel("logical-normalized class entropy")
        logical_ax.set_title(f"{name}: logical scale")
        logical_ax.grid(True, alpha=0.25)
        logical_ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()

    path = out_dir / f"{name}_exit_{suffix}_paired_derivative.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_hybrid_paired_three_panel(
    bit_result: dict,
    paired_result: dict,
    out_dir: Path,
) -> Path:
    """
    Three-panel plot using separate grids for the quantities that need them.

    The bit-scale panel uses the wider averaged curve. The derivative and
    logical panels use the paired, center-concentrated curve.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    bit_points = bit_result["points"]
    paired_points = paired_result["points"]
    bit_p = np.array([pt["p"] for pt in bit_points], dtype=float)
    paired_p = np.array([pt["p"] for pt in paired_points], dtype=float)
    name = paired_result["code"]["name"]
    suffix = "rule2" if paired_result["config"]["dual_rule2"] else "rule1"
    n = float(paired_result["code"]["n"])
    areas = paired_result.get("areas", {})

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.7))

    bit_series = [
        ("exact_x_class", "H(C_X | E,S) exact"),
        ("exact_z_class", "H(C_Z | E,S) exact"),
        ("exact_class_total", "H(C_X,C_Z | E,S) exact"),
    ]
    for key, label in bit_series:
        if key not in bit_points[0]:
            continue
        values = np.array([pt[key] for pt in bit_points], dtype=float)
        line = axes[0].plot(bit_p, values, marker=".", linewidth=1.4, label=label)[0]
        stderr_key = f"{key}_stderr"
        if stderr_key in bit_points[0]:
            stderr = np.array([pt[stderr_key] for pt in bit_points], dtype=float)
            axes[0].fill_between(
                bit_p,
                values - stderr,
                values + stderr,
                color=line.get_color(),
                alpha=0.12,
                linewidth=0,
            )
    axes[0].axhline(
        bit_result["code"]["k"],
        color="black",
        linestyle="--",
        linewidth=1,
        label="k bits",
    )
    axes[0].set_xlabel("erasure probability")
    axes[0].set_ylabel("class entropy (bits)")
    axes[0].set_title(f"{name}: bit scale")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8, loc="upper left")

    derivative_series = [
        ("exact_x_class_component_norm_dp", "d[H(C_X|E,S)/n]/dp"),
        ("exact_z_class_component_norm_dp", "d[H(C_Z|E,S)/n]/dp"),
        ("exact_class_total_norm_dp", "d[H(C_X,C_Z|E,S)/(2n)]/dp"),
    ]
    area_lines = []
    for key, label in derivative_series:
        if key not in paired_points[0]:
            continue
        values = np.array([pt[key] for pt in paired_points], dtype=float)
        line = axes[1].plot(paired_p, values, marker=".", linewidth=1.4, label=label)[0]
        area = areas.get(key, {}).get("trapezoid_derivative_area")
        if area is None:
            area = float(np.trapezoid(values, paired_p))
        area_lines.append(f"{label}: area={area:.5g} ~= {area * n:.2f}/n")
        stderr_key = f"{key}_stderr"
        if stderr_key in paired_points[0]:
            stderr = np.array([pt[stderr_key] for pt in paired_points], dtype=float)
            axes[1].fill_between(
                paired_p,
                values - stderr,
                values + stderr,
                color=line.get_color(),
                alpha=0.12,
                linewidth=0,
            )
    axes[1].set_xlabel("erasure probability")
    axes[1].set_ylabel("paired EXIT estimate d(entropy / n)/dp")
    axes[1].set_title(
        f"{name}: paired derivative scale ({len(paired_points)} p-points)"
    )
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8, loc="upper right")
    if area_lines:
        axes[1].text(
            0.02,
            0.98,
            "\n".join(area_lines),
            transform=axes[1].transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "0.75",
                "alpha": 0.9,
            },
        )

    logical_series = [
        ("exact_x_class_logical_norm", "H(C_X|E,S) / k"),
        ("exact_z_class_logical_norm", "H(C_Z|E,S) / k"),
        ("exact_class_total_logical_norm", "H(C_X,C_Z|E,S) / 2k"),
    ]
    for key, label in logical_series:
        if key not in paired_points[0]:
            continue
        values = np.array([pt[key] for pt in paired_points], dtype=float)
        line = axes[2].plot(paired_p, values, marker=".", linewidth=1.4, label=label)[0]
        stderr_key = f"{key}_stderr"
        if stderr_key in paired_points[0]:
            stderr = np.array([pt[stderr_key] for pt in paired_points], dtype=float)
            axes[2].fill_between(
                paired_p,
                values - stderr,
                values + stderr,
                color=line.get_color(),
                alpha=0.12,
                linewidth=0,
            )
    axes[2].axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=1,
        label="full logical entropy",
    )
    axes[2].set_xlabel("erasure probability")
    axes[2].set_ylabel("logical-normalized class entropy")
    axes[2].set_title(f"{name}: logical scale")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    path = out_dir / f"{name}_exit_{suffix}_hybrid_paired3.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def list_plan(code_dir: Path) -> None:
    print("Representative simulation plan:")
    for path in sorted(code_dir.glob("*.npz"), key=lambda p: p.stat().st_size):
        try:
            code = load_code(path)
        except ValueError:
            code = load_exact_class_cache(path).code
        print(
            f"{path.name:28s} n={code.n:4d} k={code.k:4d} "
            f"runs={representative_runs(code.n):4d} grid=0:0.02:1"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EXIT-style CSS erasure experiments")
    parser.add_argument("--code", type=Path, help="Path to one .npz code file")
    parser.add_argument("--codes-dir", type=Path, default=ROOT / "codes")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "experiments" / "exit_curves")
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--p-step", type=float, default=0.02)
    parser.add_argument("--ps", type=float, nargs="*", default=None)
    parser.add_argument(
        "--centered-ps",
        action="store_true",
        help="Use a nonuniform p grid concentrated around p=0.5",
    )
    parser.add_argument("--edge-step", type=float, default=0.1)
    parser.add_argument("--shoulder-step", type=float, default=0.05)
    parser.add_argument("--center-step", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exact-only", action="store_true")
    parser.add_argument("--peeling-only", action="store_true")
    parser.add_argument(
        "--exact-class-target-only",
        action="store_true",
        help="Compute only exact correction-class entropy, using packed rank queries",
    )
    parser.add_argument(
        "--exact-class-component",
        choices=["x", "z", "both"],
        default="x",
        help="Component for --exact-class-target-only",
    )
    parser.add_argument(
        "--no-logical-rank-class",
        action="store_true",
        help="Disable Lx/Lz logical-rank formula and use kernel/stabilizer ranks",
    )
    parser.add_argument("--rule2", action="store_true", help="Also use known-degree-1 dual peeling")
    parser.add_argument(
        "--paired-derivative",
        action="store_true",
        help="Estimate derivative curves by averaging paired finite differences first",
    )
    parser.add_argument(
        "--surface-fast-x-class",
        action="store_true",
        help="Fast paired derivative for repo-layout surface codes; computes only exact H(C_X|E,S)",
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plan", action="store_true", help="Print representative run counts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plan:
        list_plan(args.codes_dir)
        return
    if args.code is None:
        raise SystemExit("Provide --code, or use --plan to list representative settings.")

    skip_initial_load = bool(args.paired_derivative and args.surface_fast_x_class)
    if args.exact_class_target_only:
        n = code_length_from_npz(args.code)
        runs = args.runs if args.runs is not None else representative_runs(n)
    elif skip_initial_load and args.runs is not None:
        runs = args.runs
    else:
        code = load_code(args.code)
        runs = args.runs if args.runs is not None else representative_runs(code.n)
    if args.ps is not None and len(args.ps):
        ps = args.ps
    elif args.centered_ps:
        ps = centered_ps(
            edge_step=args.edge_step,
            shoulder_step=args.shoulder_step,
            center_step=args.center_step,
        )
    else:
        ps = default_ps(args.p_step)
    if args.paired_derivative:
        if args.surface_fast_x_class:
            result = run_surface_paired_derivative_code(
                args.code,
                ps,
                runs,
                seed=args.seed,
                use_rule2=args.rule2,
            )
        elif args.exact_class_target_only:
            result = run_paired_exact_class_target_code(
                args.code,
                ps,
                runs,
                seed=args.seed,
                component=args.exact_class_component,
                use_logicals=not args.no_logical_rank_class,
                use_rule2=args.rule2,
            )
        else:
            result = run_paired_derivative_code(
                args.code,
                ps,
                runs,
                seed=args.seed,
                include_exact=not args.peeling_only,
                include_peeling=not args.exact_only,
                use_rule2=args.rule2,
            )
        json_path, csv_path = write_paired_derivative_outputs(result, args.out_dir)
        print(f"wrote {json_path}")
        print(f"wrote {csv_path}")
        if args.plot:
            plot_path = plot_paired_derivative_result(result, args.out_dir)
            print(f"wrote {plot_path}")
        print(f"elapsed_seconds={result['elapsed_seconds']:.3f}")
        return

    if args.exact_class_target_only:
        result = run_exact_class_target_code(
            args.code,
            ps,
            runs,
            seed=args.seed,
            component=args.exact_class_component,
            use_logicals=not args.no_logical_rank_class,
            use_rule2=args.rule2,
        )
        json_path, csv_path = write_outputs(result, args.out_dir)
        print(f"wrote {json_path}")
        print(f"wrote {csv_path}")
        if args.plot:
            plot_path = plot_result(result, args.out_dir)
            print(f"wrote {plot_path}")
        print(f"elapsed_seconds={result['elapsed_seconds']:.3f}")
        return

    result = run_code(
        args.code,
        ps,
        runs,
        seed=args.seed,
        include_exact=not args.peeling_only,
        include_peeling=not args.exact_only,
        use_rule2=args.rule2,
    )
    json_path, csv_path = write_outputs(result, args.out_dir)
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    if args.plot:
        plot_path = plot_result(result, args.out_dir)
        print(f"wrote {plot_path}")
    print(f"elapsed_seconds={result['elapsed_seconds']:.3f}")


if __name__ == "__main__":
    main()
