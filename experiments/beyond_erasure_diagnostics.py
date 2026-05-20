from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.exit_curve_experiments import (
    CodeData,
    ROOT,
    code_length_from_npz,
    has_sparse_coo_matrix,
    load_code,
    npz_scalar,
    rank_gf2,
)


@dataclass(frozen=True)
class PosteriorResult:
    error_entropy: float
    class_entropy: float
    degeneracy_gain: float
    method: str
    affine_dimension: int
    list_size: int
    acceptance_rate: float | None


@dataclass(frozen=True)
class LogWeightModel:
    base_log: float
    log_odds: tuple[float, ...]
    stochastic_mask: int
    forbidden_one_mask: int
    forced_one_mask: int
    uniform_log_odds: float | None = None

    @classmethod
    def from_probabilities(cls, p_one: np.ndarray) -> "LogWeightModel":
        base_log = 0.0
        log_odds = [0.0] * int(p_one.shape[0])
        stochastic_mask = 0
        forbidden_one_mask = 0
        forced_one_mask = 0
        for col, prob in enumerate(p_one):
            bit = 1 << col
            if prob == 0.0:
                forbidden_one_mask |= bit
            elif prob == 1.0:
                forced_one_mask |= bit
            else:
                stochastic_mask |= bit
                base_log += math.log1p(-float(prob))
                log_odds[col] = math.log(float(prob)) - math.log1p(-float(prob))
        stochastic_odds = [log_odds[col] for col in range(len(log_odds)) if (stochastic_mask >> col) & 1]
        uniform_log_odds = None
        if stochastic_odds and all(
            odds == stochastic_odds[0] for odds in stochastic_odds[1:]
        ):
            uniform_log_odds = stochastic_odds[0]
        return cls(
            base_log=base_log,
            log_odds=tuple(log_odds),
            stochastic_mask=stochastic_mask,
            forbidden_one_mask=forbidden_one_mask,
            forced_one_mask=forced_one_mask,
            uniform_log_odds=uniform_log_odds,
        )

    def log_weight(self, value: int) -> float:
        if value & self.forbidden_one_mask:
            return -math.inf
        if (~value) & self.forced_one_mask:
            return -math.inf
        return self.base_log + self.sum_log_odds(value & self.stochastic_mask)

    def delta_for_toggle(self, current: int, toggle: int) -> float:
        changed = toggle & self.stochastic_mask
        turning_on = changed & ~current
        turning_off = changed & current
        return self.sum_log_odds(turning_on) - self.sum_log_odds(turning_off)

    def sum_log_odds(self, mask: int) -> float:
        if self.uniform_log_odds is not None:
            return int(mask).bit_count() * self.uniform_log_odds
        total = 0.0
        value = int(mask)
        while value:
            bit = value & -value
            total += self.log_odds[bit.bit_length() - 1]
            value ^= bit
        return total


def materialize_dense_code(path: Path, *, max_n: int) -> CodeData:
    """Load dense Hx/Hz, converting small sparse COO code files when needed."""
    n = code_length_from_npz(path)
    if n > max_n:
        raise ValueError(
            f"{path} has n={n}, above --max-dense-n={max_n}; "
            "use a smaller code or raise the limit intentionally."
        )

    with np.load(path, allow_pickle=True) as data:
        if "Hx" in data and "Hz" in data:
            return load_code(path)
        if not (has_sparse_coo_matrix(data, "HX") and has_sparse_coo_matrix(data, "HZ")):
            raise ValueError(f"{path} must contain dense Hx/Hz or sparse HX/HZ arrays")

        hx = dense_from_sparse(data, "HX")
        hz = dense_from_sparse(data, "HZ")
        lx = dense_from_sparse(data, "LX") if has_sparse_coo_matrix(data, "LX") else None
        lz = dense_from_sparse(data, "LZ") if has_sparse_coo_matrix(data, "LZ") else None

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


def dense_from_sparse(data: np.lib.npyio.NpzFile, prefix: str) -> np.ndarray:
    rows = np.asarray(data[f"rows_{prefix}"], dtype=np.int64)
    cols = np.asarray(data[f"cols_{prefix}"], dtype=np.int64)
    n_rows = npz_scalar(data, f"n_{prefix}")
    n_cols = npz_scalar(data, f"m_{prefix}")
    matrix = np.zeros((n_rows, n_cols), dtype=np.uint8)
    for row, col in zip(rows, cols):
        matrix[int(row), int(col)] ^= 1
    return matrix


def int_from_vector(vector: np.ndarray) -> int:
    value = 0
    for col in np.flatnonzero(vector % 2):
        value |= 1 << int(col)
    return value


def int_rows_from_matrix(matrix: np.ndarray) -> tuple[int, ...]:
    return tuple(int_from_vector(row) for row in (matrix % 2).astype(np.uint8))


def add_int_row_to_basis(value: int, basis: dict[int, int]) -> bool:
    while value:
        pivot = value.bit_length() - 1
        row = basis.get(pivot)
        if row is None:
            basis[pivot] = value
            return True
        value ^= row
    return False


def rowspace_basis(rows: Iterable[int]) -> dict[int, int]:
    basis: dict[int, int] = {}
    for row in rows:
        add_int_row_to_basis(int(row), basis)
    return basis


def reduce_mod_rowspace(value: int, basis: dict[int, int]) -> int:
    remainder = int(value)
    for pivot in sorted(basis, reverse=True):
        if (remainder >> pivot) & 1:
            remainder ^= basis[pivot]
    return remainder


def gf2_rref(
    matrix: np.ndarray,
    rhs: np.ndarray | None = None,
) -> tuple[np.ndarray, list[int], np.ndarray | None]:
    a = (matrix.copy() % 2).astype(np.uint8)
    if rhs is None:
        b = None
    else:
        b = (rhs.copy() % 2).astype(np.uint8)
        if b.ndim != 1 or b.shape[0] != a.shape[0]:
            raise ValueError("rhs must be a 1D vector with one entry per row")

    m, n = a.shape
    pivots: list[int] = []
    row = 0
    for col in range(n):
        candidates = np.flatnonzero(a[row:, col])
        if candidates.size == 0:
            continue
        pivot = int(candidates[0] + row)
        if pivot != row:
            a[[row, pivot]] = a[[pivot, row]]
            if b is not None:
                b[[row, pivot]] = b[[pivot, row]]

        for other in range(m):
            if other != row and a[other, col]:
                a[other] ^= a[row]
                if b is not None:
                    b[other] ^= b[row]

        pivots.append(col)
        row += 1
        if row == m:
            break

    if b is not None:
        for idx in range(row, m):
            if not a[idx].any() and b[idx]:
                raise ValueError("inconsistent GF(2) linear system")

    return a[:row], pivots, b[:row] if b is not None else None


def affine_solution_space(
    checks: np.ndarray,
    syndrome: np.ndarray,
    p_one: np.ndarray,
) -> tuple[int, tuple[int, ...]]:
    """Return one solution and a nullspace basis after deterministic priors."""
    n = checks.shape[1]
    deterministic_rows = []
    deterministic_rhs = []
    for col, prob in enumerate(p_one):
        if prob == 0.0 or prob == 1.0:
            row = np.zeros(n, dtype=np.uint8)
            row[col] = 1
            deterministic_rows.append(row)
            deterministic_rhs.append(1 if prob == 1.0 else 0)

    if deterministic_rows:
        matrix = np.vstack([checks, np.vstack(deterministic_rows)]).astype(np.uint8)
        rhs = np.concatenate(
            [syndrome.astype(np.uint8), np.asarray(deterministic_rhs, dtype=np.uint8)]
        )
    else:
        matrix = checks.astype(np.uint8)
        rhs = syndrome.astype(np.uint8)

    rref, pivots, reduced_rhs = gf2_rref(matrix, rhs)
    if reduced_rhs is None:
        raise ValueError("internal error: missing reduced rhs")

    solution = np.zeros(n, dtype=np.uint8)
    for row_idx, pivot_col in enumerate(pivots):
        solution[pivot_col] = reduced_rhs[row_idx]

    pivot_set = set(pivots)
    basis: list[int] = []
    for free_col in range(n):
        if free_col in pivot_set:
            continue
        vector = np.zeros(n, dtype=np.uint8)
        vector[free_col] = 1
        for row_idx, pivot_col in enumerate(pivots):
            if rref[row_idx, free_col]:
                vector[pivot_col] = 1
        basis.append(int_from_vector(vector))

    return int_from_vector(solution), tuple(basis)


def entropy_from_log_weights(log_weights: list[float]) -> float:
    finite = [w for w in log_weights if math.isfinite(w)]
    if not finite:
        return 0.0
    max_log = max(finite)
    weights = np.exp(np.asarray(finite, dtype=float) - max_log)
    probs = weights / weights.sum()
    log_probs = np.log(probs)
    return float(-(probs @ log_probs) / math.log(2.0))


def logaddexp_dict(accumulator: dict[int, float], key: int, value: float) -> None:
    old = accumulator.get(key)
    accumulator[key] = value if old is None else float(np.logaddexp(old, value))


def enumerate_affine_space(particular: int, basis: tuple[int, ...]) -> Iterable[int]:
    state = int(particular)
    previous_gray = 0
    yield state
    for idx in range(1, 1 << len(basis)):
        gray = idx ^ (idx >> 1)
        changed = gray ^ previous_gray
        basis_idx = changed.bit_length() - 1
        state ^= basis[basis_idx]
        previous_gray = gray
        yield state


def exact_posterior_entropy(
    particular: int,
    basis: tuple[int, ...],
    quotient_basis: dict[int, int],
    weight_model: LogWeightModel,
) -> tuple[float, float, int]:
    error_logs: list[float] = []
    class_logs: dict[int, float] = {}
    count = 0
    value = int(particular)
    lw = weight_model.log_weight(value)
    previous_gray = 0
    for idx in range(1 << len(basis)):
        if idx:
            gray = idx ^ (idx >> 1)
            changed = gray ^ previous_gray
            basis_idx = changed.bit_length() - 1
            toggle = basis[basis_idx]
            lw += weight_model.delta_for_toggle(value, toggle)
            value ^= toggle
            previous_gray = gray
        if not math.isfinite(lw):
            continue
        count += 1
        error_logs.append(lw)
        class_key = reduce_mod_rowspace(value, quotient_basis)
        logaddexp_dict(class_logs, class_key, lw)
    return entropy_from_log_weights(error_logs), entropy_from_log_weights(list(class_logs.values())), count


def random_affine_state(
    particular: int,
    basis: tuple[int, ...],
    rng: np.random.Generator,
) -> int:
    state = int(particular)
    if not basis:
        return state
    toggles = rng.integers(0, 2, size=len(basis), dtype=np.uint8)
    for idx in np.flatnonzero(toggles):
        state ^= basis[int(idx)]
    return state


def mcmc_list_entropy(
    particular: int,
    basis: tuple[int, ...],
    quotient_basis: dict[int, int],
    weight_model: LogWeightModel,
    rng: np.random.Generator,
    *,
    samples: int,
    burnin: int,
    thin: int,
) -> tuple[float, float, int, float]:
    current = int(particular)
    current_log = weight_model.log_weight(current)
    for _ in range(1000):
        if math.isfinite(current_log):
            break
        current = random_affine_state(particular, basis, rng)
        current_log = weight_model.log_weight(current)
    if not math.isfinite(current_log):
        raise ValueError("could not find a finite posterior state for MCMC")

    unique: set[int] = set()
    accepted = 0
    proposals = 0
    total_steps = int(burnin + samples * max(1, thin))
    proposal_indices = (
        rng.integers(0, len(basis), size=total_steps) if basis else np.zeros(total_steps, dtype=np.int64)
    )
    accept_logs = np.log(rng.random(total_steps))
    for step in range(total_steps):
        if basis:
            toggle = basis[int(proposal_indices[step])]
            proposal = current ^ toggle
            proposal_log = current_log + weight_model.delta_for_toggle(current, toggle)
        else:
            proposal = current
            proposal_log = current_log
        proposals += 1
        log_accept = proposal_log - current_log
        if log_accept >= 0.0 or float(accept_logs[step]) < log_accept:
            current = proposal
            current_log = proposal_log
            accepted += 1

        if step >= burnin and (step - burnin) % max(1, thin) == 0:
            unique.add(current)

    error_logs: list[float] = []
    class_logs: dict[int, float] = {}
    for value in unique:
        lw = weight_model.log_weight(value)
        if not math.isfinite(lw):
            continue
        error_logs.append(lw)
        class_key = reduce_mod_rowspace(value, quotient_basis)
        logaddexp_dict(class_logs, class_key, lw)

    acceptance = accepted / proposals if proposals else 0.0
    return (
        entropy_from_log_weights(error_logs),
        entropy_from_log_weights(list(class_logs.values())),
        len(unique),
        acceptance,
    )


def posterior_entropy_for_sample(
    checks: np.ndarray,
    quotient_rows: np.ndarray,
    syndrome: np.ndarray,
    p_one: np.ndarray,
    rng: np.random.Generator,
    *,
    max_exact_affine_dim: int,
    approx_samples: int,
    mcmc_burnin: int,
    mcmc_thin: int,
) -> PosteriorResult:
    particular, basis = affine_solution_space(checks, syndrome, p_one)
    quotient_basis = rowspace_basis(int_rows_from_matrix(quotient_rows))
    weight_model = LogWeightModel.from_probabilities(p_one)
    affine_dim = len(basis)

    if affine_dim <= max_exact_affine_dim:
        error_entropy, class_entropy, count = exact_posterior_entropy(
            particular,
            basis,
            quotient_basis,
            weight_model,
        )
        return PosteriorResult(
            error_entropy=error_entropy,
            class_entropy=class_entropy,
            degeneracy_gain=error_entropy - class_entropy,
            method="exact_affine_enumeration",
            affine_dimension=affine_dim,
            list_size=count,
            acceptance_rate=None,
        )

    if approx_samples <= 0:
        raise ValueError(
            f"affine dimension {affine_dim} exceeds --max-exact-affine-dim="
            f"{max_exact_affine_dim}; set --approx-samples for list approximation"
        )

    error_entropy, class_entropy, list_size, acceptance = mcmc_list_entropy(
        particular,
        basis,
        quotient_basis,
        weight_model,
        rng,
        samples=approx_samples,
        burnin=mcmc_burnin,
        thin=mcmc_thin,
    )
    return PosteriorResult(
        error_entropy=error_entropy,
        class_entropy=class_entropy,
        degeneracy_gain=error_entropy - class_entropy,
        method="mcmc_restricted_list",
        affine_dimension=affine_dim,
        list_size=list_size,
        acceptance_rate=acceptance,
    )


def sample_component_error(
    rng: np.random.Generator,
    p_one: np.ndarray,
) -> np.ndarray:
    return (rng.random(p_one.shape[0]) < p_one).astype(np.uint8)


def component_matrices(code: CodeData, component: str) -> tuple[np.ndarray, np.ndarray, str]:
    if component == "x":
        return code.hz, code.hx, "S_Z"
    if component == "z":
        return code.hx, code.hz, "S_X"
    raise ValueError(f"unsupported component: {component}")


def stderr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(np.asarray(values, dtype=float), ddof=1) / math.sqrt(len(values)))


def summarize_samples(
    samples: list[PosteriorResult],
    *,
    component: str,
    n: int,
    k: int,
) -> dict[str, float | int | str]:
    prefix = f"posterior_{component}"
    error_values = [sample.error_entropy for sample in samples]
    class_values = [sample.class_entropy for sample in samples]
    gain_values = [sample.degeneracy_gain for sample in samples]
    affine_values = [sample.affine_dimension for sample in samples]
    list_values = [sample.list_size for sample in samples]
    accept_values = [
        sample.acceptance_rate for sample in samples if sample.acceptance_rate is not None
    ]
    exact_count = sum(1 for sample in samples if sample.method == "exact_affine_enumeration")
    list_count = sum(1 for sample in samples if sample.method == "mcmc_restricted_list")

    error_mean = float(np.mean(error_values)) if error_values else 0.0
    class_mean = float(np.mean(class_values)) if class_values else 0.0
    gain_mean = float(np.mean(gain_values)) if gain_values else 0.0
    out: dict[str, float | int | str] = {
        "valid_runs": len(samples),
        "exact_runs": exact_count,
        "list_runs": list_count,
        f"{prefix}_error": error_mean,
        f"{prefix}_error_stderr": stderr(error_values),
        f"{prefix}_class": class_mean,
        f"{prefix}_class_stderr": stderr(class_values),
        f"{prefix}_degeneracy_gain": gain_mean,
        f"{prefix}_degeneracy_gain_stderr": stderr(gain_values),
        f"{prefix}_error_component_norm": error_mean / n,
        f"{prefix}_class_component_norm": class_mean / n,
        f"{prefix}_degeneracy_gain_component_norm": gain_mean / n,
        "mean_affine_dimension": float(np.mean(affine_values)) if affine_values else 0.0,
        "max_affine_dimension": int(max(affine_values)) if affine_values else 0,
        "mean_list_size": float(np.mean(list_values)) if list_values else 0.0,
    }
    if k > 0:
        out[f"{prefix}_class_logical_norm"] = class_mean / k
        out[f"{prefix}_class_logical_norm_stderr"] = stderr(class_values) / k
    if accept_values:
        out["mean_mcmc_acceptance_rate"] = float(np.mean(accept_values))
    return out


def run_grid(
    code: CodeData,
    *,
    component: str,
    p_erasure_grid: list[float],
    p_error_grid: list[float],
    runs: int,
    seed: int,
    max_exact_affine_dim: int,
    approx_samples: int,
    mcmc_burnin: int,
    mcmc_thin: int,
) -> dict:
    checks, quotient_rows, syndrome_name = component_matrices(code, component)
    rng = np.random.default_rng(seed)
    points: list[dict] = []
    start = time.perf_counter()

    for p_erasure in p_erasure_grid:
        for p_error in p_error_grid:
            sample_results: list[PosteriorResult] = []
            for _ in range(runs):
                erased = rng.random(code.n) < p_erasure
                p_one = np.where(erased, 0.5, p_error).astype(float)
                error = sample_component_error(rng, p_one)
                syndrome = (checks @ error % 2).astype(np.uint8)
                sample_results.append(
                    posterior_entropy_for_sample(
                        checks,
                        quotient_rows,
                        syndrome,
                        p_one,
                        rng,
                        max_exact_affine_dim=max_exact_affine_dim,
                        approx_samples=approx_samples,
                        mcmc_burnin=mcmc_burnin,
                        mcmc_thin=mcmc_thin,
                    )
                )

            point = {
                "p_erasure": float(p_erasure),
                "p_error": float(p_error),
                "runs": int(runs),
                **summarize_samples(
                    sample_results,
                    component=component,
                    n=code.n,
                    k=code.k,
                ),
            }
            points.append(point)

    return {
        "code": {
            "path": str(code.path),
            "name": code.name,
            "n": code.n,
            "rank_hx": code.rank_hx,
            "rank_hz": code.rank_hz,
            "k": code.k,
            "rate": code.k / code.n if code.n else 0.0,
        },
        "config": {
            "component": component,
            "syndrome": syndrome_name,
            "runs": runs,
            "seed": seed,
            "p_erasure_grid": p_erasure_grid,
            "p_error_grid": p_error_grid,
            "max_exact_affine_dim": max_exact_affine_dim,
            "approx_samples": approx_samples,
            "mcmc_burnin": mcmc_burnin,
            "mcmc_thin": mcmc_thin,
        },
        "quantity_notation": {
            f"posterior_{component}_error": {
                "notation": f"H({component} | Y,S)",
                "description": "Posterior component-error entropy in bits. Exact only when all samples use exact_affine_enumeration.",
            },
            f"posterior_{component}_class": {
                "notation": f"H(C_{component.upper()} | Y,S)",
                "description": "Posterior correction-class entropy modulo same-type stabilizers. MCMC rows are list-restricted estimates.",
            },
            f"posterior_{component}_degeneracy_gain": {
                "notation": f"H({component} | Y,S) - H(C_{component.upper()} | Y,S)",
                "description": "Entropy reduction from stabilizer degeneracy under the posterior model.",
            },
        },
        "points": points,
        "elapsed_seconds": time.perf_counter() - start,
    }


def output_stem(result: dict) -> str:
    component = result["config"]["component"]
    return f"{result['code']['name']}_beyond_erasure_{component}"


def write_outputs(result: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(result)
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"

    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    fieldnames = sorted({key for point in result["points"] for key in point})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for point in result["points"]:
            writer.writerow(point)
    return json_path, csv_path


def plot_result(result: dict, out_dir: Path) -> Path:
    points = result["points"]
    component = result["config"]["component"]
    prefix = f"posterior_{component}"
    p_erasure = sorted({float(point["p_erasure"]) for point in points})
    p_error = sorted({float(point["p_error"]) for point in points})
    by_pair = {(float(point["p_erasure"]), float(point["p_error"])): point for point in points}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    if len(p_erasure) > 1 and len(p_error) > 1:
        class_grid = np.array(
            [
                [by_pair[(pe, px)].get(f"{prefix}_class_logical_norm", 0.0) for px in p_error]
                for pe in p_erasure
            ],
            dtype=float,
        )
        gain_grid = np.array(
            [
                [
                    by_pair[(pe, px)][f"{prefix}_degeneracy_gain_component_norm"]
                    for px in p_error
                ]
                for pe in p_erasure
            ],
            dtype=float,
        )
        extent = [min(p_error), max(p_error), min(p_erasure), max(p_erasure)]
        im0 = axes[0].imshow(
            class_grid,
            origin="lower",
            aspect="auto",
            extent=extent,
            interpolation="nearest",
        )
        axes[0].set_title(f"H(C_{component.upper()} | Y,S) / k")
        axes[0].set_xlabel("Pauli component error probability")
        axes[0].set_ylabel("erasure probability")
        fig.colorbar(im0, ax=axes[0], fraction=0.046)

        im1 = axes[1].imshow(
            gain_grid,
            origin="lower",
            aspect="auto",
            extent=extent,
            interpolation="nearest",
        )
        axes[1].set_title("degeneracy gain / n")
        axes[1].set_xlabel("Pauli component error probability")
        axes[1].set_ylabel("erasure probability")
        fig.colorbar(im1, ax=axes[1], fraction=0.046)
    else:
        x_key = "p_error" if len(p_error) > 1 else "p_erasure"
        x_label = "Pauli component error probability" if x_key == "p_error" else "erasure probability"
        xs = np.array([float(point[x_key]) for point in points], dtype=float)
        order = np.argsort(xs)
        xs = xs[order]
        class_values = np.array(
            [points[int(idx)].get(f"{prefix}_class_logical_norm", 0.0) for idx in order],
            dtype=float,
        )
        gain_values = np.array(
            [points[int(idx)][f"{prefix}_degeneracy_gain_component_norm"] for idx in order],
            dtype=float,
        )
        axes[0].plot(xs, class_values, marker=".")
        axes[0].set_title(f"H(C_{component.upper()} | Y,S) / k")
        axes[0].set_xlabel(x_label)
        axes[0].set_ylabel("logical-normalized class entropy")
        axes[0].grid(True, alpha=0.25)

        axes[1].plot(xs, gain_values, marker=".")
        axes[1].set_title("degeneracy gain / n")
        axes[1].set_xlabel(x_label)
        axes[1].set_ylabel("component-normalized bits")
        axes[1].grid(True, alpha=0.25)

    fig.suptitle(f"{result['code']['name']}: beyond-erasure {component.upper()} diagnostic")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{output_stem(result)}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def parse_probability_grid(values: list[float] | None, default: list[float]) -> list[float]:
    if not values:
        return default
    grid = sorted({round(float(value), 12) for value in values})
    for value in grid:
        if value < 0.0 or value > 1.0:
            raise ValueError("probability grids must stay within [0, 1]")
    return grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EXIT/GEXIT-style posterior class diagnostics beyond pure erasure"
    )
    parser.add_argument("--code", type=Path, required=True, help="Path to a CSS .npz code")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "experiments" / "beyond_erasure",
    )
    parser.add_argument("--component", choices=["x", "z"], default="x")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p-erasure-grid", type=float, nargs="*", default=None)
    parser.add_argument("--p-error-grid", type=float, nargs="*", default=None)
    parser.add_argument("--max-dense-n", type=int, default=1200)
    parser.add_argument("--max-exact-affine-dim", type=int, default=20)
    parser.add_argument(
        "--approx-samples",
        type=int,
        default=2000,
        help="MCMC sample count for list-restricted estimates when exact enumeration is too large",
    )
    parser.add_argument("--mcmc-burnin", type=int, default=1000)
    parser.add_argument("--mcmc-thin", type=int, default=2)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    p_erasure_grid = parse_probability_grid(args.p_erasure_grid, [0.0, 0.1, 0.2, 0.3])
    p_error_grid = parse_probability_grid(args.p_error_grid, [0.0, 0.01, 0.03, 0.05])
    code = materialize_dense_code(args.code, max_n=args.max_dense_n)
    result = run_grid(
        code,
        component=args.component,
        p_erasure_grid=p_erasure_grid,
        p_error_grid=p_error_grid,
        runs=args.runs,
        seed=args.seed,
        max_exact_affine_dim=args.max_exact_affine_dim,
        approx_samples=args.approx_samples,
        mcmc_burnin=args.mcmc_burnin,
        mcmc_thin=args.mcmc_thin,
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
