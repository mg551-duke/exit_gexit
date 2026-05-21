from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from bsc_gexit_surface_sampled import (  # noqa: E402
    Factor,
    axis_key,
    binary_rank,
    bits_to_int,
    centered_axis_values,
    code_plot_label,
    column_supports,
    contract_factors,
    format_scale,
    min_fill_order,
)
from exit_curve_experiments import load_code  # noqa: E402


DEFAULT_OUT_DIR = ROOT / "data" / "experiments" / "depolarizing_gexit_curves"

_WORKER_HZ_SUPPORTS: tuple[tuple[int, ...], ...] | None = None
_WORKER_HX_SUPPORTS: tuple[tuple[int, ...], ...] | None = None
_WORKER_LZ_SUPPORTS: tuple[tuple[int, ...], ...] | None = None
_WORKER_LX_SUPPORTS: tuple[tuple[int, ...], ...] | None = None
_WORKER_SYNDROME_MODEL: "DepolarizingFactorModel | None" = None
_WORKER_QUOTIENT_MODEL: "DepolarizingFactorModel | None" = None
_WORKER_SYNDROME_RANK: int | None = None
_WORKER_QUOTIENT_RANK: int | None = None
_WORKER_N: int | None = None
_WORKER_P_GRID: np.ndarray | None = None


@dataclass(frozen=True)
class DepolarizingFactorModel:
    n_constraints: int
    rank: int
    grouped_scopes: tuple[tuple[tuple[int, ...], tuple[int, ...], int], ...]
    elimination_order: tuple[int, ...]

    @classmethod
    def from_xz_matrices(
        cls,
        x_matrix: np.ndarray,
        z_matrix: np.ndarray,
    ) -> "DepolarizingFactorModel":
        x_matrix = (x_matrix % 2).astype(np.uint8)
        z_matrix = (z_matrix % 2).astype(np.uint8)
        if x_matrix.shape != z_matrix.shape:
            raise ValueError("x and z observation matrices must have the same shape")

        pairs: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        for col in range(x_matrix.shape[1]):
            x_scope = tuple(int(row) for row in np.flatnonzero(x_matrix[:, col]))
            z_scope = tuple(int(row) for row in np.flatnonzero(z_matrix[:, col]))
            if x_scope or z_scope:
                pairs.append((x_scope, z_scope))

        grouped = tuple(
            (x_scope, z_scope, count)
            for (x_scope, z_scope), count in sorted(Counter(pairs).items())
        )
        order = min_fill_order(
            [tuple(sorted(set(x_scope) | set(z_scope))) for x_scope, z_scope in pairs],
            x_matrix.shape[0],
        )
        matrix = np.concatenate([x_matrix, z_matrix], axis=1)
        return cls(
            n_constraints=int(x_matrix.shape[0]),
            rank=binary_rank(matrix),
            grouped_scopes=grouped,
            elimination_order=tuple(order),
        )

    def probability(self, rhs: np.ndarray, p: float) -> float:
        if p == 0.0:
            return 1.0 if not np.any(rhs.astype(np.uint8) % 2) else 0.0
        tau = 1.0 - 4.0 * p / 3.0
        factors: list[Factor] = []
        for x_scope, z_scope, count in self.grouped_scopes:
            scope = tuple(sorted(set(x_scope) | set(z_scope)))
            factors.append(
                Factor(
                    scope=scope,
                    table=depolarizing_character_table(
                        x_scope,
                        z_scope,
                        scope,
                        tau**count,
                    ),
                )
            )

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
        description="Coupled sampled depolarizing GEXIT curves for surface codes."
    )
    parser.add_argument("--distance", type=int, default=5)
    parser.add_argument("--code", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=max(1, min(os.cpu_count() or 1, 4)))
    parser.add_argument("--seed", type=int, default=5005)
    parser.add_argument("--ps", nargs="*", type=float, default=None)
    parser.add_argument(
        "--channel-entropy-max",
        type=float,
        default=2.0,
        help=(
            "Maximum raw depolarizing entropy H4(p). The default spans the "
            "full Pauli channel, so the normalized entropy axis H4(p)/2 runs "
            "from 0 to 1."
        ),
    )
    parser.add_argument("--entropy-edge-step", type=float, default=0.1)
    parser.add_argument("--entropy-shoulder-step", type=float, default=0.05)
    parser.add_argument("--entropy-center-step", type=float, default=0.02)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def depolarizing_character_table(
    x_scope: tuple[int, ...],
    z_scope: tuple[int, ...],
    joint_scope: tuple[int, ...],
    nonzero_value: float,
) -> np.ndarray:
    if not joint_scope:
        return np.asarray(1.0)
    positions = {var: idx for idx, var in enumerate(joint_scope)}
    x_positions = [positions[var] for var in x_scope]
    z_positions = [positions[var] for var in z_scope]
    table = np.empty((2,) * len(joint_scope), dtype=np.float64)
    for index in np.ndindex(table.shape):
        x_parity = sum(index[pos] for pos in x_positions) % 2
        z_parity = sum(index[pos] for pos in z_positions) % 2
        table[index] = 1.0 if x_parity == 0 and z_parity == 0 else nonzero_value
    return table


def binary_entropy(prob: float) -> float:
    if prob <= 0.0 or prob >= 1.0:
        return 0.0
    return -prob * math.log2(prob) - (1.0 - prob) * math.log2(1.0 - prob)


def depolarizing_entropy(p: float) -> float:
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return math.log2(3.0)
    return binary_entropy(p) + p * math.log2(3.0)


def depolarizing_entropy_axis(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    values = np.zeros_like(p)
    mask = p > 0.0
    values[mask] = (
        -p[mask] * np.log2(p[mask])
        - (1.0 - p[mask]) * np.log2(1.0 - p[mask])
        + p[mask] * np.log2(3.0)
    )
    return values


def normalized_depolarizing_entropy_axis(p: np.ndarray) -> np.ndarray:
    return 0.5 * depolarizing_entropy_axis(p)


def depolarizing_entropy_derivative(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    derivative = np.full_like(p, np.nan)
    derivative[p == 0.0] = np.inf
    mask = (p > 0.0) & (p < 0.75)
    derivative[mask] = np.log2(3.0 * (1.0 - p[mask]) / p[mask])
    return derivative


def inverse_depolarizing_entropy(target: float) -> float:
    if target <= 0.0:
        return 0.0
    if target >= 2.0:
        return 0.75
    lo = 0.0
    hi = 0.75
    for _ in range(90):
        mid = 0.5 * (lo + hi)
        if depolarizing_entropy(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def entropy_centered_depolarizing_ps(
    *,
    channel_entropy_max: float,
    edge_step: float,
    shoulder_step: float,
    center_step: float,
) -> list[float]:
    if channel_entropy_max <= 0.0 or channel_entropy_max > 2.0:
        raise ValueError("channel_entropy_max must lie in (0, 2]")
    return [
        inverse_depolarizing_entropy(channel_entropy_max * value)
        for value in centered_axis_values(
            edge_step=edge_step,
            shoulder_step=shoulder_step,
            center_step=center_step,
        )
    ]


def distance_from_name(name: str) -> int | None:
    match = re.search(r"surface(\d+)", name)
    return int(match.group(1)) if match else None


def code_key(name: str, distance: int | None) -> str:
    if distance is not None:
        return f"surface{distance}"
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


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


def progress_bar(completed: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    filled = min(width, int(round(width * completed / total)))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def log_progress(label: str, completed: int, total: int) -> None:
    pct = 100.0 * completed / total if total else 100.0
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(
        f"[{timestamp}] {label} {completed}/{total} "
        f"{pct:5.1f}% {progress_bar(completed, total)}",
        flush=True,
    )


def observation_matrices(
    *,
    hx: np.ndarray,
    hz: np.ndarray,
    lx: np.ndarray,
    lz: np.ndarray,
    include_logicals: bool,
) -> tuple[np.ndarray, np.ndarray]:
    n = hx.shape[1]
    syndrome_x = np.hstack([hz, np.zeros((hz.shape[0], n), dtype=np.uint8)])
    syndrome_z = np.hstack([np.zeros((hx.shape[0], n), dtype=np.uint8), hx])
    rows = [syndrome_x, syndrome_z]
    if include_logicals:
        logical_x = np.hstack([lz, np.zeros((lz.shape[0], n), dtype=np.uint8)])
        logical_z = np.hstack([np.zeros((lx.shape[0], n), dtype=np.uint8), lx])
        rows.extend([logical_x, logical_z])
    combined = np.vstack(rows).astype(np.uint8)
    return combined[:, :n], combined[:, n:]


def apply_pauli_to_state(
    *,
    qubit: int,
    label: int,
    syndrome: np.ndarray,
    logical: np.ndarray,
    hz_supports: tuple[tuple[int, ...], ...],
    hx_supports: tuple[tuple[int, ...], ...],
    lz_supports: tuple[tuple[int, ...], ...],
    lx_supports: tuple[tuple[int, ...], ...],
    syndrome_z_offset: int,
    logical_z_offset: int,
) -> None:
    has_x = label in (0, 2)
    has_z = label in (1, 2)
    if has_x:
        for check in hz_supports[qubit]:
            syndrome[check] ^= 1
        for row in lz_supports[qubit]:
            logical[row] ^= 1
    if has_z:
        for check in hx_supports[qubit]:
            syndrome[syndrome_z_offset + check] ^= 1
        for row in lx_supports[qubit]:
            logical[logical_z_offset + row] ^= 1


def estimate_batch(
    *,
    hz_supports: tuple[tuple[int, ...], ...],
    hx_supports: tuple[tuple[int, ...], ...],
    lz_supports: tuple[tuple[int, ...], ...],
    lx_supports: tuple[tuple[int, ...], ...],
    syndrome_model: DepolarizingFactorModel,
    quotient_model: DepolarizingFactorModel,
    syndrome_rank: int,
    quotient_rank: int,
    n: int,
    p_grid: np.ndarray,
    samples: int,
    rng: np.random.Generator,
    progress_label: str | None = None,
) -> dict[str, np.ndarray | int]:
    point_count = int(p_grid.size)
    syndrome_z_offset = syndrome_rank // 2
    logical_rank = quotient_rank - syndrome_rank
    logical_z_offset = logical_rank // 2
    sum_h_s = np.zeros(point_count, dtype=np.float64)
    sumsq_h_s = np.zeros(point_count, dtype=np.float64)
    sum_class = np.zeros(point_count, dtype=np.float64)
    sumsq_class = np.zeros(point_count, dtype=np.float64)
    sum_dydp = np.zeros(point_count, dtype=np.float64)
    sumsq_dydp = np.zeros(point_count, dtype=np.float64)
    syndrome_caches: list[dict[int, float]] = [dict() for _ in range(point_count)]
    quotient_caches: list[dict[int, float]] = [dict() for _ in range(point_count)]
    stride = max(1, samples // 20)
    if progress_label is not None:
        log_progress(progress_label, 0, samples)

    for sample_idx in range(1, samples + 1):
        thresholds = rng.random(n)
        pauli_labels = rng.integers(0, 3, size=n, dtype=np.uint8)
        order = np.argsort(thresholds)
        next_flip = 0
        syndrome = np.zeros(syndrome_rank, dtype=np.uint8)
        logical = np.zeros(logical_rank, dtype=np.uint8)
        class_values = np.empty(point_count, dtype=np.float64)
        h_s_values = np.empty(point_count, dtype=np.float64)
        for idx, p in enumerate(p_grid):
            while next_flip < n and thresholds[order[next_flip]] < p:
                qubit = int(order[next_flip])
                apply_pauli_to_state(
                    qubit=qubit,
                    label=int(pauli_labels[qubit]),
                    syndrome=syndrome,
                    logical=logical,
                    hz_supports=hz_supports,
                    hx_supports=hx_supports,
                    lz_supports=lz_supports,
                    lx_supports=lx_supports,
                    syndrome_z_offset=syndrome_z_offset,
                    logical_z_offset=logical_z_offset,
                )
                next_flip += 1

            s_key = bits_to_int(syndrome)
            p_s = syndrome_caches[idx].get(s_key)
            if p_s is None:
                p_s = syndrome_model.probability(syndrome, float(p))
                syndrome_caches[idx][s_key] = p_s

            quotient_rhs = np.concatenate([syndrome, logical])
            q_key = bits_to_int(quotient_rhs)
            p_q = quotient_caches[idx].get(q_key)
            if p_q is None:
                p_q = quotient_model.probability(quotient_rhs, float(p))
                quotient_caches[idx][q_key] = p_q

            h_s = -math.log2(max(p_s, np.finfo(float).tiny))
            h_q = -math.log2(max(p_q, np.finfo(float).tiny))
            h_s_values[idx] = h_s
            class_values[idx] = h_q - h_s

        class_norm = class_values / n
        dydp = np.gradient(class_norm, p_grid)
        dydp[0] = 0.0
        dydp[-1] = 0.0
        sum_h_s += h_s_values
        sumsq_h_s += h_s_values * h_s_values
        sum_class += class_values
        sumsq_class += class_values * class_values
        sum_dydp += dydp
        sumsq_dydp += dydp * dydp
        if progress_label is not None and (sample_idx % stride == 0 or sample_idx == samples):
            log_progress(progress_label, sample_idx, samples)

    return {
        "samples": int(samples),
        "sum_h_s": sum_h_s,
        "sumsq_h_s": sumsq_h_s,
        "sum_class": sum_class,
        "sumsq_class": sumsq_class,
        "sum_dydp": sum_dydp,
        "sumsq_dydp": sumsq_dydp,
    }


def init_worker(
    hz_supports: tuple[tuple[int, ...], ...],
    hx_supports: tuple[tuple[int, ...], ...],
    lz_supports: tuple[tuple[int, ...], ...],
    lx_supports: tuple[tuple[int, ...], ...],
    syndrome_model: DepolarizingFactorModel,
    quotient_model: DepolarizingFactorModel,
    syndrome_rank: int,
    quotient_rank: int,
    n: int,
    p_grid: np.ndarray,
) -> None:
    global _WORKER_HZ_SUPPORTS
    global _WORKER_HX_SUPPORTS
    global _WORKER_LZ_SUPPORTS
    global _WORKER_LX_SUPPORTS
    global _WORKER_SYNDROME_MODEL
    global _WORKER_QUOTIENT_MODEL
    global _WORKER_SYNDROME_RANK
    global _WORKER_QUOTIENT_RANK
    global _WORKER_N
    global _WORKER_P_GRID
    _WORKER_HZ_SUPPORTS = hz_supports
    _WORKER_HX_SUPPORTS = hx_supports
    _WORKER_LZ_SUPPORTS = lz_supports
    _WORKER_LX_SUPPORTS = lx_supports
    _WORKER_SYNDROME_MODEL = syndrome_model
    _WORKER_QUOTIENT_MODEL = quotient_model
    _WORKER_SYNDROME_RANK = syndrome_rank
    _WORKER_QUOTIENT_RANK = quotient_rank
    _WORKER_N = n
    _WORKER_P_GRID = np.asarray(p_grid, dtype=float)


def estimate_batch_worker(task: tuple[int, int, int, int, bool]) -> dict[str, np.ndarray | int]:
    batch_idx, batch_count, samples, seed, progress = task
    if (
        _WORKER_HZ_SUPPORTS is None
        or _WORKER_HX_SUPPORTS is None
        or _WORKER_LZ_SUPPORTS is None
        or _WORKER_LX_SUPPORTS is None
        or _WORKER_SYNDROME_MODEL is None
        or _WORKER_QUOTIENT_MODEL is None
        or _WORKER_SYNDROME_RANK is None
        or _WORKER_QUOTIENT_RANK is None
        or _WORKER_N is None
        or _WORKER_P_GRID is None
    ):
        raise RuntimeError("worker is not initialized")
    return estimate_batch(
        hz_supports=_WORKER_HZ_SUPPORTS,
        hx_supports=_WORKER_HX_SUPPORTS,
        lz_supports=_WORKER_LZ_SUPPORTS,
        lx_supports=_WORKER_LX_SUPPORTS,
        syndrome_model=_WORKER_SYNDROME_MODEL,
        quotient_model=_WORKER_QUOTIENT_MODEL,
        syndrome_rank=_WORKER_SYNDROME_RANK,
        quotient_rank=_WORKER_QUOTIENT_RANK,
        n=_WORKER_N,
        p_grid=_WORKER_P_GRID,
        samples=samples,
        rng=np.random.default_rng(seed),
        progress_label=f"batch {batch_idx + 1}/{batch_count}" if progress else None,
    )


def merge_batches(
    batches: list[dict[str, np.ndarray | int]],
    p_grid: np.ndarray,
    n: int,
) -> list[dict[str, float]]:
    samples = int(sum(int(batch["samples"]) for batch in batches))
    totals = {
        name: sum(
            (batch[name] for batch in batches),
            np.zeros_like(p_grid, dtype=np.float64),
        )
        for name in ("sum_h_s", "sumsq_h_s", "sum_class", "sumsq_class", "sum_dydp", "sumsq_dydp")
    }
    mean_h_s = totals["sum_h_s"] / samples
    mean_class = totals["sum_class"] / samples
    mean_dydp = np.maximum(totals["sum_dydp"] / samples, 0.0)
    h_s_stderr = standard_error(totals["sum_h_s"], totals["sumsq_h_s"], samples)
    class_stderr = standard_error(totals["sum_class"], totals["sumsq_class"], samples)
    dydp_stderr = standard_error(totals["sum_dydp"], totals["sumsq_dydp"], samples)

    rows = []
    for idx, p in enumerate(p_grid):
        raw_error = n * depolarizing_entropy(float(p)) - mean_h_s[idx]
        saved = raw_error - mean_class[idx]
        rows.append(
            {
                "p": float(p),
                "channel_entropy": depolarizing_entropy(float(p)),
                "normalized_channel_entropy": depolarizing_entropy(float(p)) / 2.0,
                "posterior_pauli_error": float(raw_error),
                "posterior_logical_class": float(mean_class[idx]),
                "posterior_saved_by_stabilizers": float(saved),
                "posterior_pauli_error_stderr": float(h_s_stderr[idx]),
                "posterior_logical_class_stderr": float(class_stderr[idx]),
                "posterior_logical_class_component_norm": float(mean_class[idx] / n),
                "posterior_logical_class_component_norm_stderr": float(class_stderr[idx] / n),
                "posterior_logical_class_component_norm_dp": float(mean_dydp[idx]),
                "posterior_logical_class_component_norm_dp_stderr": float(dydp_stderr[idx]),
            }
        )
    return rows


def resolve_p_grid(args: argparse.Namespace) -> np.ndarray:
    if args.ps is not None:
        p_values = [float(p) for p in args.ps]
    else:
        p_values = entropy_centered_depolarizing_ps(
            channel_entropy_max=args.channel_entropy_max,
            edge_step=args.entropy_edge_step,
            shoulder_step=args.entropy_shoulder_step,
            center_step=args.entropy_center_step,
        )
    rounded = sorted({round(float(p), 15) for p in p_values})
    if len(rounded) < 2:
        raise ValueError("p grid must contain at least two points")
    if rounded[0] < 0.0 or rounded[-1] > 0.75:
        raise ValueError("default depolarizing entropy grid must stay within [0, 0.75]")
    return np.array(rounded, dtype=float)


def compute_result(
    code_path: Path,
    *,
    p_grid: np.ndarray,
    samples: int,
    seed: int,
    workers: int,
    progress: bool = True,
) -> dict:
    code = load_code(code_path)
    if code.lx is None or code.lz is None:
        raise ValueError("depolarizing GEXIT requires Lx and Lz logicals")

    syndrome_x, syndrome_z = observation_matrices(
        hx=code.hx,
        hz=code.hz,
        lx=code.lx,
        lz=code.lz,
        include_logicals=False,
    )
    quotient_x, quotient_z = observation_matrices(
        hx=code.hx,
        hz=code.hz,
        lx=code.lx,
        lz=code.lz,
        include_logicals=True,
    )
    syndrome_model = DepolarizingFactorModel.from_xz_matrices(syndrome_x, syndrome_z)
    quotient_model = DepolarizingFactorModel.from_xz_matrices(quotient_x, quotient_z)
    syndrome_rank = syndrome_x.shape[0]
    quotient_rank = quotient_x.shape[0]
    hz_supports = column_supports(code.hz)
    hx_supports = column_supports(code.hx)
    lz_supports = column_supports(code.lz)
    lx_supports = column_supports(code.lx)

    workers = max(1, int(workers))
    samples = int(samples)
    batch_count = min(workers, samples)
    base_count, remainder = divmod(samples, batch_count)
    sample_counts = [base_count + (1 if idx < remainder else 0) for idx in range(batch_count)]
    seed_sequence = np.random.SeedSequence(seed)
    batch_seeds = [
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in seed_sequence.spawn(batch_count)
    ]
    tasks = [
        (idx, batch_count, int(sample_count), int(batch_seed), bool(progress))
        for idx, (sample_count, batch_seed) in enumerate(zip(sample_counts, batch_seeds))
    ]
    if progress:
        print(
            f"starting depolarizing GEXIT: d={distance_from_name(code.name)}, "
            f"{samples} samples, {p_grid.size} p-points, {batch_count} batches",
            flush=True,
        )

    if batch_count == 1:
        batches = [
            estimate_batch(
                hz_supports=hz_supports,
                hx_supports=hx_supports,
                lz_supports=lz_supports,
                lx_supports=lx_supports,
                syndrome_model=syndrome_model,
                quotient_model=quotient_model,
                syndrome_rank=syndrome_rank,
                quotient_rank=quotient_rank,
                n=code.n,
                p_grid=p_grid,
                samples=tasks[0][2],
                rng=np.random.default_rng(tasks[0][3]),
                progress_label="batch 1/1" if progress else None,
            )
        ]
    else:
        with ProcessPoolExecutor(
            max_workers=batch_count,
            initializer=init_worker,
            initargs=(
                hz_supports,
                hx_supports,
                lz_supports,
                lx_supports,
                syndrome_model,
                quotient_model,
                syndrome_rank,
                quotient_rank,
                code.n,
                p_grid,
            ),
        ) as executor:
            batches = list(executor.map(estimate_batch_worker, tasks))

    rows = merge_batches(batches, p_grid, code.n)
    dydp = np.array(
        [row["posterior_logical_class_component_norm_dp"] for row in rows],
        dtype=float,
    )
    peak_idx = int(np.argmax(dydp))
    peak = float(dydp[peak_idx])
    scale = 1.0 / peak if peak > 0.0 else 1.0
    for row in rows:
        value = row["posterior_logical_class_component_norm_dp"]
        stderr = row["posterior_logical_class_component_norm_dp_stderr"]
        row["scaled_posterior_logical_class_component_norm_dp"] = float(value * scale)
        row["scaled_posterior_logical_class_component_norm_dp_stderr"] = float(
            stderr * scale
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
        },
        "channel": {
            "name": "depolarizing",
            "parameter": "total non-identity Pauli probability p",
            "probabilities": {"I": "1-p", "X": "p/3", "Y": "p/3", "Z": "p/3"},
            "entropy": "h2(p)+p*log2(3)",
            "normalized_entropy": "(h2(p)+p*log2(3))/2",
        },
        "method": "coupled sampled exact-probability contraction",
        "samples": samples,
        "seed": seed,
        "workers": workers,
        "batch_count": batch_count,
        "grid": {
            "p": [float(p) for p in p_grid],
            "channel_entropy": [float(t) for t in depolarizing_entropy_axis(p_grid)],
            "normalized_channel_entropy": [
                float(t) for t in normalized_depolarizing_entropy_axis(p_grid)
            ],
        },
        "observation": {
            "syndrome_rank": syndrome_rank,
            "quotient_rank": quotient_rank,
            "logical_rank": quotient_rank - syndrome_rank,
        },
        "scaling": {
            "peak_p": float(rows[peak_idx]["p"]),
            "peak_channel_entropy": float(rows[peak_idx]["channel_entropy"]),
            "peak_normalized_channel_entropy": float(
                rows[peak_idx]["normalized_channel_entropy"]
            ),
            "peak_gexit": peak,
            "scale_to_unit_peak": scale,
            "trapezoid_area_component_norm": float(np.trapezoid(dydp, p_grid)),
            "target_full_logical_area_2k_over_n": 2.0 * code.k / code.n,
        },
        "points": rows,
    }


def transformed_gexit_values(result: dict) -> tuple[np.ndarray, np.ndarray, float]:
    p = np.array([point["p"] for point in result["points"]], dtype=float)
    dydp = np.array(
        [point["posterior_logical_class_component_norm_dp"] for point in result["points"]],
        dtype=float,
    )
    hprime = depolarizing_entropy_derivative(p)
    # Entropy-axis panels use t = H4(p)/2 in [0, 1], so d/dt = 2 d/dH4.
    dydt = np.divide(
        2.0 * dydp,
        hprime,
        out=np.full_like(dydp, np.nan),
        where=hprime > 0.0,
    )
    dydt[np.isclose(p, 0.75)] = 0.0
    finite = np.isfinite(dydt)
    peak = float(np.max(dydt[finite])) if np.any(finite) else 0.0
    return normalized_depolarizing_entropy_axis(p), dydt, peak


def tikz_coordinates(points: list[dict], key: str, x_key: str = "p") -> str:
    return " ".join(f"({point[x_key]:.12g},{point[key]:.12g})" for point in points)


def write_outputs(result: dict, out_dir: Path) -> None:
    code = result["code"]
    key = code_key(code["name"], code["distance"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tikz_dir = out_dir / "tikz"
    tikz_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{key}_depolarizing_gexit_sampled.json"
    csv_path = out_dir / f"{key}_depolarizing_gexit_sampled.csv"
    bit_png = out_dir / f"{key}_depolarizing_bit_scale_sampled.png"
    bit_entropy_png = out_dir / f"{key}_depolarizing_bit_scale_entropy_axis_sampled.png"
    gexit_png = out_dir / f"{key}_scaled_depolarizing_gexit_sampled.png"
    gexit_entropy_png = out_dir / f"{key}_scaled_depolarizing_gexit_entropy_axis_sampled.png"
    gexit_tikz = tikz_dir / f"fig_scaled_depolarizing_gexit_{key}_sampled_compact.tex"

    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    fields = list(result["points"][0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["points"])

    plot_bit_scale(result, bit_png, entropy_axis=False)
    plot_bit_scale(result, bit_entropy_png, entropy_axis=True)
    plot_scaled_gexit(result, gexit_png)
    plot_scaled_gexit_entropy_axis(result, gexit_entropy_png)
    write_gexit_tikz(result, gexit_tikz)

    for path in (
        json_path,
        csv_path,
        bit_png,
        bit_entropy_png,
        gexit_png,
        gexit_entropy_png,
        gexit_tikz,
    ):
        print(f"wrote {path}")


def plot_bit_scale(result: dict, out_path: Path, *, entropy_axis: bool) -> None:
    points = result["points"]
    x_key = "normalized_channel_entropy" if entropy_axis else "p"
    x = np.array([point[x_key] for point in points], dtype=float)
    series = [
        ("posterior_pauli_error", r"$\mathbb{E}H(E\mid S)$"),
        ("posterior_logical_class", r"$\mathbb{E}H(L\mid S)$"),
        ("posterior_saved_by_stabilizers", r"$\mathbb{E}H(E\mid L,S)$"),
    ]
    fig, ax = plt.subplots(figsize=(6.1, 4.1))
    for key, label in series:
        ax.plot(x, [point[key] for point in points], marker=".", linewidth=1.35, label=label)
    ax.axhline(result["code"]["k"] * 2, color="black", linestyle="--", linewidth=0.9, label=r"$2k$")
    ax.set_xlabel(r"normalized depolarizing entropy $H_4(p)/2$" if entropy_axis else r"depolarizing probability $p$")
    ax.set_ylabel("expected entropy (bits)")
    ax.set_title("Sampled depolarizing bit scale")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_scaled_gexit(result: dict, out_path: Path) -> None:
    points = result["points"]
    p = np.array([point["p"] for point in points], dtype=float)
    y = np.array([point["scaled_posterior_logical_class_component_norm_dp"] for point in points])
    scale = result["scaling"]["scale_to_unit_peak"]
    code = result["code"]
    fig, ax = plt.subplots(figsize=(6.1, 4.1))
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
    ax.plot(
        p,
        y,
        marker=".",
        linewidth=1.35,
        label=rf"{code_plot_label(code['name'], code['n'], code['k'], code['distance'])}, scaled by ${format_scale(scale)}$",
    )
    ax.set_xlabel(r"depolarizing probability $p$")
    ax.set_ylabel(r"$\widetilde g^{\rm depol}(p)$")
    ax.set_title("Sampled depolarizing GEXIT derivative")
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
    ax.plot(
        t[finite],
        y[finite],
        marker=".",
        linewidth=1.35,
        label=rf"{code_plot_label(code['name'], code['n'], code['k'], code['distance'])}, scaled by ${format_scale(scale)}$",
    )
    ax.set_xlabel(r"normalized depolarizing entropy $H_4(p)/2$")
    ax.set_ylabel(r"$\widetilde g^{\rm depol}(H_4/2)$")
    ax.set_title("Sampled depolarizing GEXIT derivative")
    ax.set_ylim(-0.02, 1.06)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_gexit_tikz(result: dict, out_path: Path) -> None:
    code = result["code"]
    key = code_key(code["name"], code["distance"])
    axis = axis_key(key)
    t, dydt, peak = transformed_gexit_values(result)
    scale_t = 1.0 / peak if peak > 0.0 else 1.0
    xmax = float(np.nanmax(t)) if t.size else 1.0
    lines = [
        f"% Compact PGFPlots panel for sampled {key} depolarizing GEXIT derivative.",
        "% Requires \\usepackage{pgfplots} and \\pgfplotsset{compat=1.18}.",
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        f"  name=scaledDepolarizingGEXIT{axis}Axis,",
        r"  width=\linewidth,",
        r"  height=0.78\linewidth,",
        r"  title={Depolarizing derivative},",
        r"  title style={font=\scriptsize},",
        r"  xlabel={$H_4(p)/2$},",
        r"  ylabel={$\widetilde g^{\rm depol}(H_4/2)$},",
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
        rf"\addplot[black!60, dashed, line width=0.55pt, forget plot] coordinates {{(0,1) ({xmax:.12g},1)}};",
    ]
    coords = " ".join(
        f"({x:.12g},{(y * scale_t):.12g})"
        for x, y in zip(t, dydt)
        if np.isfinite(y)
    )
    lines.extend(
        [
            r"\addplot+[color=blue, mark=*, mark size=1.0pt, line width=0.65pt]",
            rf"coordinates {{{coords}}};",
            rf"\addlegendentry{{{code_plot_label(code['name'], code['n'], code['k'], code['distance'])}, scaled by ${format_scale(scale_t)}$}}",
            "\\end{axis}",
            "\\end{tikzpicture}",
            "",
        ]
    )
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
        p_grid=resolve_p_grid(args),
        samples=args.samples,
        seed=args.seed,
        workers=args.workers,
        progress=not args.no_progress,
    )
    write_outputs(result, args.out_dir.resolve())
    scaling = result["scaling"]
    print(
        "area check: "
        f"{scaling['trapezoid_area_component_norm']:.8g} vs "
        f"2k/n={scaling['target_full_logical_area_2k_over_n']:.8g}; "
        f"peak at p={scaling['peak_p']:.4g}, "
        f"H4/2={scaling['peak_normalized_channel_entropy']:.4g}"
    )


if __name__ == "__main__":
    main()
