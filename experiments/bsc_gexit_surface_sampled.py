from __future__ import annotations

import argparse
import csv
import json
import math
import re
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exit_curve_experiments import ROOT, load_code


DEFAULT_OUT_DIR = ROOT / "data" / "experiments" / "gexit_curves"
DEFAULT_TIKZ_DIR = DEFAULT_OUT_DIR / "tikz"

_WORKER_HZ: np.ndarray | None = None
_WORKER_Q_MODEL: "FactorModel | None" = None
_WORKER_SYNDROME_RANK: int | None = None
_WORKER_QUOTIENT_RANK: int | None = None
_WORKER_N: int | None = None
_WORKER_P_GRID: np.ndarray | None = None
_WORKER_HZ_COLUMN_SUPPORTS: tuple[tuple[int, ...], ...] | None = None
_WORKER_LZ_BITS: np.ndarray | None = None


@dataclass(frozen=True)
class Factor:
    scope: tuple[int, ...]
    table: np.ndarray


@dataclass(frozen=True)
class FactorModel:
    n_constraints: int
    rank: int
    grouped_scopes: tuple[tuple[tuple[int, ...], int], ...]
    elimination_order: tuple[int, ...]

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "FactorModel":
        matrix = (matrix % 2).astype(np.uint8)
        scopes = []
        for col in range(matrix.shape[1]):
            scope = tuple(int(row) for row in np.flatnonzero(matrix[:, col]))
            if scope:
                scopes.append(scope)
        grouped = tuple(sorted(Counter(scopes).items()))
        order = min_fill_order([scope for scope, _count in grouped], matrix.shape[0])
        return cls(
            n_constraints=int(matrix.shape[0]),
            rank=binary_rank(matrix),
            grouped_scopes=grouped,
            elimination_order=tuple(order),
        )

    def probability(self, rhs: np.ndarray, p: float) -> float:
        if p == 0.5:
            return 2.0 ** (-self.rank)
        t = 1.0 - 2.0 * p
        factors: list[Factor] = []
        for scope, count in self.grouped_scopes:
            factors.append(Factor(scope=scope, table=parity_table(len(scope), t**count)))

        for idx, bit in enumerate(rhs.astype(np.uint8) % 2):
            if bit:
                factors.append(Factor(scope=(idx,), table=np.array([1.0, -1.0])))

        total = contract_factors(factors, self.elimination_order)
        prob = total / (2.0 ** self.n_constraints)
        if prob < 0.0 and prob > -1e-12:
            return 0.0
        return float(max(prob, 0.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sampled exact-probability BSC/GEXIT curves for planar surface codes."
    )
    parser.add_argument("--distance", type=int, default=7)
    parser.add_argument("--code", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tikz-dir", type=Path, default=DEFAULT_TIKZ_DIR)
    parser.add_argument("--points", type=int, default=26)
    parser.add_argument("--ps", nargs="*", type=float, default=None)
    parser.add_argument(
        "--entropy-centered-ps",
        action="store_true",
        help=(
            "Use a nonuniform grid that is centered in t=h2(p), then invert "
            "back to BSC crossover probabilities."
        ),
    )
    parser.add_argument("--entropy-edge-step", type=float, default=0.1)
    parser.add_argument("--entropy-shoulder-step", type=float, default=0.05)
    parser.add_argument("--entropy-center-step", type=float, default=0.02)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes used across coupled sample batches.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def min_fill_order(scopes: list[tuple[int, ...]], n_vars: int) -> list[int]:
    neighbors = [set() for _ in range(n_vars)]
    for scope in scopes:
        for first in scope:
            for second in scope:
                if first != second:
                    neighbors[first].add(second)

    remaining = set(range(n_vars))
    order: list[int] = []
    while remaining:
        best = None
        best_key = None
        for var in remaining:
            active = neighbors[var] & remaining
            fill = 0
            active_list = list(active)
            for i, first in enumerate(active_list):
                for second in active_list[i + 1 :]:
                    if second not in neighbors[first]:
                        fill += 1
            key = (fill, len(active), var)
            if best_key is None or key < best_key:
                best_key = key
                best = var
        assert best is not None
        active = neighbors[best] & remaining
        for first in active:
            for second in active:
                if first != second:
                    neighbors[first].add(second)
        remaining.remove(best)
        order.append(best)
    return order


def parity_table(width: int, odd_value: float) -> np.ndarray:
    table = np.empty((2,) * width, dtype=np.float64)
    for index in np.ndindex(table.shape):
        table[index] = odd_value if sum(index) % 2 else 1.0
    return table


def broadcast_factor(factor: Factor, joint_scope: tuple[int, ...]) -> np.ndarray:
    shape = [1] * len(joint_scope)
    positions = {var: idx for idx, var in enumerate(joint_scope)}
    for axis, var in enumerate(factor.scope):
        shape[positions[var]] = factor.table.shape[axis]
    return factor.table.reshape(shape)


def contract_factors(factors: list[Factor], order: tuple[int, ...]) -> float:
    active = list(factors)
    for var in order:
        bucket = [factor for factor in active if var in factor.scope]
        if not bucket:
            continue
        active = [factor for factor in active if var not in factor.scope]
        joint_scope = tuple(sorted(set().union(*(factor.scope for factor in bucket))))
        joint = np.ones((2,) * len(joint_scope), dtype=np.float64)
        for factor in bucket:
            joint *= broadcast_factor(factor, joint_scope)
        axis = joint_scope.index(var)
        reduced = joint.sum(axis=axis)
        reduced_scope = tuple(item for item in joint_scope if item != var)
        if reduced_scope:
            active.append(Factor(scope=reduced_scope, table=reduced))
        else:
            active.append(Factor(scope=(), table=np.asarray(reduced)))

    total = np.asarray(1.0)
    for factor in active:
        if factor.scope:
            total = total * factor.table.sum()
        else:
            total = total * factor.table
    return float(total)


def binary_entropy(prob: float) -> float:
    if prob <= 0.0 or prob >= 1.0:
        return 0.0
    return -prob * math.log2(prob) - (1.0 - prob) * math.log2(1.0 - prob)


def centered_axis_values(
    *,
    edge_step: float = 0.1,
    shoulder_step: float = 0.05,
    center_step: float = 0.02,
) -> list[float]:
    if edge_step <= 0.0 or shoulder_step <= 0.0 or center_step <= 0.0:
        raise ValueError("grid steps must be positive")

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


def inverse_binary_entropy(target: float) -> float:
    if target <= 0.0:
        return 0.0
    if target >= 1.0:
        return 0.5

    lo = 0.0
    hi = 0.5
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if binary_entropy(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def entropy_centered_bsc_ps(
    *,
    edge_step: float = 0.1,
    shoulder_step: float = 0.05,
    center_step: float = 0.02,
) -> list[float]:
    return [
        inverse_binary_entropy(t)
        for t in centered_axis_values(
            edge_step=edge_step,
            shoulder_step=shoulder_step,
            center_step=center_step,
        )
    ]


def resolve_p_grid(args: argparse.Namespace) -> np.ndarray:
    if args.ps is not None and args.entropy_centered_ps:
        raise ValueError("--ps and --entropy-centered-ps are mutually exclusive")
    if args.ps is not None:
        p_values = [float(p) for p in args.ps]
    elif args.entropy_centered_ps:
        p_values = entropy_centered_bsc_ps(
            edge_step=args.entropy_edge_step,
            shoulder_step=args.entropy_shoulder_step,
            center_step=args.entropy_center_step,
        )
    else:
        p_values = np.linspace(0.0, 0.5, args.points).tolist()

    rounded = sorted({round(float(p), 15) for p in p_values})
    if len(rounded) < 2:
        raise ValueError("p grid must contain at least two points")
    if rounded[0] < 0.0 or rounded[-1] > 0.5:
        raise ValueError("BSC p grid must stay within [0, 0.5]")
    return np.array(rounded, dtype=float)


def binary_entropy_axis(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    t = np.zeros_like(p)
    mask = (p > 0.0) & (p < 1.0)
    t[mask] = (
        -p[mask] * np.log2(p[mask])
        - (1.0 - p[mask]) * np.log2(1.0 - p[mask])
    )
    return t


def binary_entropy_derivative(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    derivative = np.full_like(p, np.nan)
    derivative[p == 0.0] = np.inf
    mask = (p > 0.0) & (p < 1.0)
    derivative[mask] = np.log2((1.0 - p[mask]) / p[mask])
    return derivative


def transformed_gexit_values(result: dict) -> tuple[np.ndarray, np.ndarray, float]:
    points = result["points"]
    p = np.array([point["p"] for point in points], dtype=float)
    dydp = np.array(
        [point["posterior_x_class_component_norm_dp"] for point in points],
        dtype=float,
    )
    t = binary_entropy_axis(p)
    hprime = binary_entropy_derivative(p)
    dydt = np.divide(
        dydp,
        hprime,
        out=np.full_like(dydp, np.nan),
        where=hprime > 0.0,
    )
    finite = np.isfinite(dydt)
    peak = float(np.max(dydt[finite])) if np.any(finite) else 0.0
    return t, dydt, peak


def row_to_int(row: np.ndarray) -> int:
    value = 0
    for idx in np.flatnonzero(row.astype(np.uint8) % 2):
        value |= 1 << int(idx)
    return value


def add_row_to_basis(value: int, basis: dict[int, int]) -> bool:
    value = int(value)
    while value:
        pivot = value.bit_length() - 1
        existing = basis.get(pivot)
        if existing is None:
            basis[pivot] = value
            return True
        value ^= existing
    return False


def binary_rank(matrix: np.ndarray) -> int:
    basis: dict[int, int] = {}
    rank = 0
    for row in matrix:
        if add_row_to_basis(row_to_int(row), basis):
            rank += 1
    return rank


def bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for idx, bit in enumerate(bits.astype(np.uint8) % 2):
        if bit:
            value |= 1 << idx
    return value


def column_supports(matrix: np.ndarray) -> tuple[tuple[int, ...], ...]:
    matrix = (matrix % 2).astype(np.uint8)
    return tuple(
        tuple(int(row) for row in np.flatnonzero(matrix[:, col]))
        for col in range(matrix.shape[1])
    )


def distance_from_name(name: str) -> int | None:
    match = re.search(r"surface(\d+)", name)
    return int(match.group(1)) if match else None


def code_key(name: str, n: int, distance: int | None) -> str:
    if distance is not None:
        return f"surface{distance}"
    base = name.replace("_HxHzLxLz", "")
    if base in {"gross", "two_gross"}:
        return f"{base}{n}"
    return re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_")


def code_plot_label(name: str, n: int, k: int, distance: int | None) -> str:
    if distance is not None:
        return f"$d={distance}$"
    base = name.replace("_HxHzLxLz", "").replace("_", r"\_")
    return rf"{base} $[[{n},{k}]]$"


def axis_key(key: str) -> str:
    parts = re.split(r"[^A-Za-z0-9]+", key)
    return "".join(part[:1].upper() + part[1:] for part in parts if part)


def format_scale(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) >= 1000.0 or abs(value) < 0.01:
        exponent = int(math.floor(math.log10(abs(value))))
        mantissa = value / (10.0**exponent)
        return rf"{mantissa:.3g}\times 10^{{{exponent}}}"
    return f"{value:.3g}"


def sample_neg_logs_from_state(
    *,
    q_model: FactorModel,
    syndrome_rank: int,
    quotient_rank: int,
    p: float,
    syndrome: np.ndarray,
    logical: int,
    cache: dict[int, tuple[float, float]],
) -> tuple[float, float]:
    if p == 0.0:
        return 0.0, 0.0
    if p == 0.5:
        return float(syndrome_rank), float(quotient_rank)

    key = bits_to_int(syndrome)
    cached = cache.get(key)
    if cached is None:
        rhs0 = np.concatenate([syndrome, np.array([0], dtype=np.uint8)])
        rhs1 = np.concatenate([syndrome, np.array([1], dtype=np.uint8)])
        p0 = q_model.probability(rhs0, p)
        p1 = q_model.probability(rhs1, p)
        cached = (p0, p1)
        cache[key] = cached
    p0, p1 = cached
    p_s = max(p0 + p1, np.finfo(float).tiny)
    p_sm = max(p1 if logical else p0, np.finfo(float).tiny)
    return -math.log2(p_s), -math.log2(p_sm)


def estimate_coupled_batch(
    *,
    hz: np.ndarray,
    hz_column_supports: tuple[tuple[int, ...], ...],
    lz_bits: np.ndarray,
    q_model: FactorModel,
    syndrome_rank: int,
    quotient_rank: int,
    p_grid: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray | int]:
    n = hz.shape[1]
    p_grid = np.asarray(p_grid, dtype=float)
    point_count = int(p_grid.size)
    sum_h_s = np.zeros(point_count, dtype=np.float64)
    sumsq_h_s = np.zeros(point_count, dtype=np.float64)
    sum_h_q = np.zeros(point_count, dtype=np.float64)
    sumsq_h_q = np.zeros(point_count, dtype=np.float64)
    sum_class = np.zeros(point_count, dtype=np.float64)
    sumsq_class = np.zeros(point_count, dtype=np.float64)
    sum_dydp = np.zeros(point_count, dtype=np.float64)
    sumsq_dydp = np.zeros(point_count, dtype=np.float64)
    caches: list[dict[int, tuple[float, float]]] = [dict() for _ in range(point_count)]

    for _ in range(samples):
        thresholds = rng.random(n)
        threshold_order = np.argsort(thresholds)
        next_flip = 0
        syndrome = np.zeros(hz.shape[0], dtype=np.uint8)
        logical = 0
        h_s_values = np.empty(point_count, dtype=np.float64)
        h_q_values = np.empty(point_count, dtype=np.float64)
        for idx, p in enumerate(p_grid):
            if p == 0.0:
                h_s, h_q = 0.0, 0.0
            elif p == 0.5:
                h_s, h_q = float(syndrome_rank), float(quotient_rank)
            else:
                while next_flip < n and thresholds[threshold_order[next_flip]] < p:
                    qubit = int(threshold_order[next_flip])
                    for check in hz_column_supports[qubit]:
                        syndrome[check] ^= 1
                    if lz_bits[qubit]:
                        logical ^= 1
                    next_flip += 1
                h_s, h_q = sample_neg_logs_from_state(
                    q_model=q_model,
                    syndrome_rank=syndrome_rank,
                    quotient_rank=quotient_rank,
                    p=float(p),
                    syndrome=syndrome,
                    logical=logical,
                    cache=caches[idx],
                )
            h_s_values[idx] = h_s
            h_q_values[idx] = h_q

        class_values = h_q_values - h_s_values
        class_norm_values = class_values / n
        dydp_values = np.gradient(class_norm_values, p_grid)
        dydp_values[0] = 0.0
        dydp_values[-1] = 0.0

        sum_h_s += h_s_values
        sumsq_h_s += h_s_values * h_s_values
        sum_h_q += h_q_values
        sumsq_h_q += h_q_values * h_q_values
        sum_class += class_values
        sumsq_class += class_values * class_values
        sum_dydp += dydp_values
        sumsq_dydp += dydp_values * dydp_values

    return {
        "samples": int(samples),
        "sum_h_s": sum_h_s,
        "sumsq_h_s": sumsq_h_s,
        "sum_h_q": sum_h_q,
        "sumsq_h_q": sumsq_h_q,
        "sum_class": sum_class,
        "sumsq_class": sumsq_class,
        "sum_dydp": sum_dydp,
        "sumsq_dydp": sumsq_dydp,
    }


def standard_error(
    sum_values: np.ndarray,
    sumsq_values: np.ndarray,
    samples: int,
) -> np.ndarray:
    if samples <= 1:
        return np.zeros_like(sum_values)
    variance = (sumsq_values - sum_values * sum_values / samples) / (samples - 1)
    variance = np.maximum(variance, 0.0)
    return np.sqrt(variance / samples)


def merge_coupled_batches(
    batches: list[dict[str, np.ndarray | int]],
    p_grid: np.ndarray,
    n: int,
) -> list[dict[str, float]]:
    samples = int(sum(int(batch["samples"]) for batch in batches))
    if samples <= 0:
        raise ValueError("at least one coupled sample is required")

    names = (
        "sum_h_s",
        "sumsq_h_s",
        "sum_h_q",
        "sumsq_h_q",
        "sum_class",
        "sumsq_class",
        "sum_dydp",
        "sumsq_dydp",
    )
    totals = {
        name: sum(
            (batch[name] for batch in batches),
            np.zeros_like(p_grid, dtype=np.float64),
        )
        for name in names
    }
    mean_h_s = totals["sum_h_s"] / samples
    mean_class = totals["sum_class"] / samples
    mean_dydp = totals["sum_dydp"] / samples
    h_s_stderr = standard_error(totals["sum_h_s"], totals["sumsq_h_s"], samples)
    class_stderr = standard_error(totals["sum_class"], totals["sumsq_class"], samples)
    dydp_stderr = standard_error(totals["sum_dydp"], totals["sumsq_dydp"], samples)

    mean_dydp = np.maximum(mean_dydp, 0.0)
    rows = []
    for idx, p in enumerate(p_grid):
        raw_error = n * binary_entropy(float(p)) - mean_h_s[idx]
        class_entropy = mean_class[idx]
        saved = raw_error - class_entropy
        rows.append(
            {
                "p": float(p),
                "posterior_x_error": float(raw_error),
                "posterior_x_class": float(class_entropy),
                "posterior_x_saved_by_stabilizers": float(saved),
                "posterior_x_error_stderr": float(h_s_stderr[idx]),
                "posterior_x_class_stderr": float(class_stderr[idx]),
                "posterior_x_class_component_norm": float(class_entropy / n),
                "posterior_x_class_component_norm_stderr": float(class_stderr[idx] / n),
                "posterior_x_class_component_norm_dp": float(mean_dydp[idx]),
                "posterior_x_class_component_norm_dp_stderr": float(dydp_stderr[idx]),
            }
        )
    return rows


def init_estimate_worker(
    hz: np.ndarray,
    lz: np.ndarray,
    q_model: FactorModel,
    syndrome_rank: int,
    quotient_rank: int,
    n: int,
    p_grid: np.ndarray | None = None,
) -> None:
    global _WORKER_HZ
    global _WORKER_Q_MODEL
    global _WORKER_SYNDROME_RANK
    global _WORKER_QUOTIENT_RANK
    global _WORKER_N
    global _WORKER_P_GRID
    global _WORKER_HZ_COLUMN_SUPPORTS
    global _WORKER_LZ_BITS
    _WORKER_HZ = hz
    _WORKER_Q_MODEL = q_model
    _WORKER_SYNDROME_RANK = syndrome_rank
    _WORKER_QUOTIENT_RANK = quotient_rank
    _WORKER_N = n
    _WORKER_P_GRID = None if p_grid is None else np.asarray(p_grid, dtype=float)
    _WORKER_HZ_COLUMN_SUPPORTS = column_supports(hz)
    _WORKER_LZ_BITS = (lz[0] % 2).astype(np.uint8)


def estimate_coupled_batch_worker(task: tuple[int, int]) -> dict[str, np.ndarray | int]:
    samples, point_seed = task
    if (
        _WORKER_HZ is None
        or _WORKER_Q_MODEL is None
        or _WORKER_SYNDROME_RANK is None
        or _WORKER_QUOTIENT_RANK is None
        or _WORKER_N is None
        or _WORKER_P_GRID is None
        or _WORKER_HZ_COLUMN_SUPPORTS is None
        or _WORKER_LZ_BITS is None
    ):
        raise RuntimeError("coupled estimate worker was not initialized")
    return estimate_coupled_batch(
        hz=_WORKER_HZ,
        hz_column_supports=_WORKER_HZ_COLUMN_SUPPORTS,
        lz_bits=_WORKER_LZ_BITS,
        q_model=_WORKER_Q_MODEL,
        syndrome_rank=_WORKER_SYNDROME_RANK,
        quotient_rank=_WORKER_QUOTIENT_RANK,
        p_grid=_WORKER_P_GRID,
        samples=int(samples),
        rng=np.random.default_rng(point_seed),
    )


def compute_result(
    code_path: Path,
    points: int | None = None,
    samples: int = 300,
    seed: int = 0,
    p_grid: np.ndarray | None = None,
    workers: int = 1,
) -> dict:
    code = load_code(code_path)
    if code.lz is None:
        raise ValueError("surface sampled BSC experiment requires Lz in the .npz")
    hz = code.hz
    lz = code.lz
    q_matrix = np.vstack([hz, lz])
    q_model = FactorModel.from_matrix(q_matrix)
    syndrome_rank = binary_rank(hz)
    quotient_rank = q_model.rank
    hz_column_supports = column_supports(hz)
    lz_bits = (lz[0] % 2).astype(np.uint8)
    if p_grid is None:
        if points is None:
            raise ValueError("either points or p_grid must be provided")
        p_grid = np.linspace(0.0, 0.5, points)
    else:
        p_grid = np.array(sorted({round(float(p), 15) for p in p_grid}), dtype=float)
    if p_grid.size < 2:
        raise ValueError("p grid must contain at least two points")
    if p_grid[0] < 0.0 or p_grid[-1] > 0.5:
        raise ValueError("BSC p grid must stay within [0, 0.5]")
    samples = int(samples)
    if samples <= 0:
        raise ValueError("samples must be positive")
    workers = max(1, int(workers))
    batch_count = min(workers, samples)
    base_count, remainder = divmod(samples, batch_count)
    sample_counts = [
        base_count + (1 if idx < remainder else 0)
        for idx in range(batch_count)
    ]
    seed_sequence = np.random.SeedSequence(seed)
    batch_seeds = [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in seed_sequence.spawn(batch_count)
    ]
    tasks = [
        (int(sample_count), int(batch_seed))
        for sample_count, batch_seed in zip(sample_counts, batch_seeds)
    ]
    if batch_count == 1:
        batches = [
            estimate_coupled_batch(
                hz=hz,
                hz_column_supports=hz_column_supports,
                lz_bits=lz_bits,
                q_model=q_model,
                syndrome_rank=syndrome_rank,
                quotient_rank=quotient_rank,
                p_grid=p_grid,
                samples=tasks[0][0],
                rng=np.random.default_rng(tasks[0][1]),
            )
        ]
    else:
        with ProcessPoolExecutor(
            max_workers=batch_count,
            initializer=init_estimate_worker,
            initargs=(hz, lz, q_model, syndrome_rank, quotient_rank, code.n, p_grid),
        ) as executor:
            batches = list(executor.map(estimate_coupled_batch_worker, tasks))

    rows = merge_coupled_batches(batches, p_grid, code.n)
    dydp = np.array(
        [row["posterior_x_class_component_norm_dp"] for row in rows],
        dtype=float,
    )
    peak_idx = int(np.argmax(dydp))
    peak_value = float(dydp[peak_idx])
    scale = 1.0 / peak_value if peak_value > 0.0 else 1.0
    for row, value in zip(rows, dydp):
        row["scaled_posterior_x_class_component_norm_dp"] = float(value * scale)
        row["scaled_posterior_x_class_component_norm_dp_stderr"] = float(
            row["posterior_x_class_component_norm_dp_stderr"] * scale
        )

    return {
        "code": {
            "path": str(code_path),
            "name": code.name,
            "distance": distance_from_name(code.name),
            "n": code.n,
            "k": code.k,
            "rank_hx": code.rank_hx,
            "rank_hz": code.rank_hz,
            "rank_hz_observed": syndrome_rank,
            "rank_hz_lz_observed": quotient_rank,
        },
        "method": "coupled sampled exact-probability contraction",
        "samples": samples,
        "seed": seed,
        "workers": workers,
        "batch_count": batch_count,
        "paired_derivative": True,
        "grid": {
            "p": [float(p) for p in p_grid],
            "t": [float(t) for t in binary_entropy_axis(p_grid)],
        },
        "scaling": {
            "peak_p": float(rows[peak_idx]["p"]),
            "peak_gexit": peak_value,
            "scale_to_unit_peak": scale,
            "trapezoid_area_component_norm": float(np.trapezoid(dydp, p_grid)),
            "target_area_k_over_n": code.k / code.n,
        },
        "points": rows,
    }


def tikz_coordinates(points: list[dict], key: str) -> str:
    return " ".join(f"({point['p']:.12g},{point[key]:.12g})" for point in points)


def write_outputs(result: dict, out_dir: Path, tikz_dir: Path) -> None:
    code = result["code"]
    key = code_key(code["name"], code["n"], code["distance"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tikz_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{key}_bsc_gexit_sampled.json"
    csv_path = out_dir / f"{key}_bsc_gexit_sampled.csv"
    bit_png = out_dir / f"{key}_bsc_bit_scale_sampled.png"
    bit_entropy_png = out_dir / f"{key}_bsc_bit_scale_entropy_axis_sampled.png"
    gexit_png = out_dir / f"{key}_scaled_bsc_gexit_sampled.png"
    gexit_entropy_png = out_dir / f"{key}_scaled_bsc_gexit_entropy_axis_sampled.png"
    bit_tikz = tikz_dir / f"fig_{key}_bsc_bit_scale_sampled_compact.tex"
    gexit_tikz = tikz_dir / f"fig_scaled_bsc_gexit_{key}_sampled_compact.tex"

    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    fields = list(result["points"][0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["points"])

    plot_bit_scale(result, bit_png)
    plot_bit_scale_entropy_axis(result, bit_entropy_png)
    plot_scaled_gexit(result, gexit_png)
    plot_scaled_gexit_entropy_axis(result, gexit_entropy_png)
    write_bit_tikz(result, bit_tikz)
    write_gexit_tikz(result, gexit_tikz)

    for path in (
        json_path,
        csv_path,
        bit_png,
        bit_entropy_png,
        gexit_png,
        gexit_entropy_png,
        bit_tikz,
        gexit_tikz,
    ):
        print(f"wrote {path}")


def plot_bit_scale(result: dict, out_path: Path) -> None:
    points = result["points"]
    p = np.array([point["p"] for point in points])
    code = result["code"]
    series = [
        ("posterior_x_error", r"$\mathbb{E}H(x\mid S)$"),
        ("posterior_x_class", r"$\mathbb{E}H(C_X\mid S)$"),
        ("posterior_x_saved_by_stabilizers", r"$\mathbb{E}H(x\mid C_X,S)$"),
    ]
    fig, ax = plt.subplots(figsize=(6.1, 4.1))
    for key, label in series:
        ax.plot(p, [point[key] for point in points], marker=".", linewidth=1.45, label=label)
    ax.axhline(code["k"], color="black", linestyle="--", linewidth=0.9, label=r"$k$")
    ax.set_xlabel(r"BSC crossover probability $p$")
    ax.set_ylabel("expected entropy (bits)")
    title_label = code["name"].replace("_HxHzLxLz", "").replace("_", " ")
    ax.set_title(rf"{title_label}: sampled BSC bit scale")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_bit_scale_entropy_axis(result: dict, out_path: Path) -> None:
    points = result["points"]
    p = np.array([point["p"] for point in points], dtype=float)
    t = binary_entropy_axis(p)
    code = result["code"]
    series = [
        ("posterior_x_error", r"$\mathbb{E}H(x\mid S)$"),
        ("posterior_x_class", r"$\mathbb{E}H(C_X\mid S)$"),
        ("posterior_x_saved_by_stabilizers", r"$\mathbb{E}H(x\mid C_X,S)$"),
    ]
    fig, ax = plt.subplots(figsize=(6.1, 4.1))
    for key, label in series:
        ax.plot(
            t,
            [point[key] for point in points],
            marker=".",
            linewidth=1.45,
            label=label,
        )
    ax.axhline(code["k"], color="black", linestyle="--", linewidth=0.9, label=r"$k$")
    ax.set_xlabel(r"BSC channel entropy $t=h_2(p)$")
    ax.set_ylabel("expected entropy (bits)")
    title_label = code["name"].replace("_HxHzLxLz", "").replace("_", " ")
    ax.set_title(rf"{title_label}: sampled BSC bit scale")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_scaled_gexit(result: dict, out_path: Path) -> None:
    points = result["points"]
    p = np.array([point["p"] for point in points])
    y = np.array([point["scaled_posterior_x_class_component_norm_dp"] for point in points])
    code = result["code"]
    scale = result["scaling"]["scale_to_unit_peak"]
    fig, ax = plt.subplots(figsize=(6.1, 4.1))
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
    ax.axvline(0.5, color="black", linestyle=":", linewidth=0.9)
    ax.plot(
        p,
        y,
        marker=".",
        linewidth=1.45,
        label=rf"{code_plot_label(code['name'], code['n'], code['k'], code['distance'])}, scaled by ${format_scale(scale)}$",
    )
    ax.set_xlabel(r"BSC crossover probability $p$")
    ax.set_ylabel(r"$\widetilde g_X^{\rm BSC}(p)$")
    ax.set_title("Sampled BSC GEXIT derivative")
    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(-0.02, 1.06)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_scaled_gexit_entropy_axis(result: dict, out_path: Path) -> None:
    code = result["code"]
    t, dydt, peak = transformed_gexit_values(result)
    scale = 1.0 / peak if peak > 0.0 else 1.0
    y = dydt * scale
    finite = np.isfinite(y)

    fig, ax = plt.subplots(figsize=(6.1, 4.1))
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
    ax.axvline(0.5, color="black", linestyle=":", linewidth=0.9)
    ax.plot(
        t[finite],
        y[finite],
        marker=".",
        linewidth=1.45,
        label=(
            rf"{code_plot_label(code['name'], code['n'], code['k'], code['distance'])}, "
            rf"scaled by ${format_scale(scale)}$"
        ),
    )
    ax.set_xlabel(r"BSC channel entropy $t=h_2(p)$")
    ax.set_ylabel(r"$\widetilde g_X^{\rm BSC}(t)$")
    ax.set_title("Sampled BSC GEXIT derivative")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.02, 1.06)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_bit_tikz(result: dict, out_path: Path) -> None:
    points = result["points"]
    code = result["code"]
    key = code_key(code["name"], code["n"], code["distance"])
    axis = axis_key(key)
    ymax = max(point["posterior_x_error"] for point in points) * 1.05
    lines = [
        f"% Compact PGFPlots panel for sampled {key} BSC bit-scale decomposition.",
        "% Requires \\usepackage{pgfplots} and \\pgfplotsset{compat=1.18}.",
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        f"  name={axis}BSCBitScaleAxis,",
        r"  width=\linewidth,",
        r"  height=0.78\linewidth,",
        r"  title={BSC bit-scale decomposition},",
        r"  title style={font=\scriptsize},",
        r"  xlabel={$p$},",
        r"  ylabel={bits},",
        "  xmin=0, xmax=0.5,",
        f"  ymin=-0.5, ymax={ymax:.12g},",
        "  grid=both,",
        r"  major grid style={black!12},",
        r"  minor grid style={black!6},",
        "  tick align=outside,",
        r"  tick label style={font=\scriptsize},",
        r"  label style={font=\scriptsize},",
        "  legend cell align={left},",
        r"  legend style={draw=black!15, fill=white, fill opacity=0.82, text opacity=1, at={(0.02,0.98)}, anchor=north west, font=\tiny},",
        "]",
        rf"\addplot[black!70, dashed, line width=0.55pt] coordinates {{(0,{code['k']}) (0.5,{code['k']})}};",
        r"\addlegendentry{$k$}",
    ]
    series = [
        ("posterior_x_error", r"$\mathbb{E}H(x\mid S)$", "blue", "*"),
        ("posterior_x_class", r"$\mathbb{E}H(C_X\mid S)$", "orange", "square*"),
        (
            "posterior_x_saved_by_stabilizers",
            r"$\mathbb{E}H(x\mid C_X,S)$",
            "green!60!black",
            "triangle*",
        ),
    ]
    for key, label, color, marker in series:
        lines.extend(
            [
                rf"\addplot+[color={color}, mark={marker}, mark size=1.0pt, line width=0.65pt]",
                rf"coordinates {{{tikz_coordinates(points, key)}}};",
                rf"\addlegendentry{{{label}}}",
            ]
        )
    lines.extend(["\\end{axis}", "\\end{tikzpicture}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_gexit_tikz(result: dict, out_path: Path) -> None:
    code = result["code"]
    key = code_key(code["name"], code["n"], code["distance"])
    axis = axis_key(key)
    scale = result["scaling"]["scale_to_unit_peak"]
    lines = [
        f"% Compact PGFPlots panel for sampled {key} BSC GEXIT derivative.",
        "% Requires \\usepackage{pgfplots} and \\pgfplotsset{compat=1.18}.",
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        f"  name=scaledBSCGEXIT{axis}Axis,",
        r"  width=\linewidth,",
        r"  height=0.78\linewidth,",
        r"  title={BSC surface derivative},",
        r"  title style={font=\scriptsize},",
        r"  xlabel={$p$},",
        r"  ylabel={$\widetilde g_X^{\rm BSC}(p)$},",
        "  xmin=0, xmax=0.5,",
        "  ymin=-0.02, ymax=1.06,",
        "  grid=both,",
        r"  major grid style={black!12},",
        r"  minor grid style={black!6},",
        "  tick align=outside,",
        r"  tick label style={font=\scriptsize},",
        r"  label style={font=\scriptsize},",
        "  legend cell align={left},",
        r"  legend style={draw=black!15, fill=white, fill opacity=0.82, text opacity=1, at={(0.02,0.02)}, anchor=south west, font=\tiny},",
        "]",
        r"\addplot[black!60, densely dotted, line width=0.55pt, forget plot] coordinates {(0.5,-0.02) (0.5,1.06)};",
        r"\addplot[black!60, dashed, line width=0.55pt, forget plot] coordinates {(0,1) (0.5,1)};",
        r"\addplot+[color=blue, mark=*, mark size=1.0pt, line width=0.65pt]",
        rf"coordinates {{{tikz_coordinates(result['points'], 'scaled_posterior_x_class_component_norm_dp')}}};",
        rf"\addlegendentry{{{code_plot_label(code['name'], code['n'], code['k'], code['distance'])}, scaled by ${format_scale(scale)}$}}",
        "\\end{axis}",
        "\\end{tikzpicture}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    code_path = (
        args.code.resolve()
        if args.code is not None
        else ROOT / "codes" / f"surface{args.distance}_HxHzLxLz.npz"
    )
    result = compute_result(
        code_path,
        points=args.points,
        samples=args.samples,
        seed=args.seed,
        p_grid=resolve_p_grid(args),
        workers=args.workers,
    )
    write_outputs(result, args.out_dir.resolve(), args.tikz_dir.resolve())
    scaling = result["scaling"]
    print(
        "area check: "
        f"{scaling['trapezoid_area_component_norm']:.8g} vs "
        f"k/n={scaling['target_area_k_over_n']:.8g}; "
        f"peak at p={scaling['peak_p']:.4g}"
    )


if __name__ == "__main__":
    main()
