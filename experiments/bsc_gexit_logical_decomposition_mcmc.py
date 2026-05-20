from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.beyond_erasure_diagnostics import (
    LogWeightModel,
    affine_solution_space,
    entropy_from_log_weights,
    int_rows_from_matrix,
    logaddexp_dict,
    random_affine_state,
)
from experiments.exit_curve_experiments import ROOT, load_code


DEFAULT_CODE = ROOT / "codes" / "gross_HxHzLxLz.npz"
DEFAULT_OUT_DIR = ROOT / "data" / "experiments" / "gexit_curves"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decompose sampled BSC correction-class entropy into logical-basis "
            "chain-rule contributions using posterior MCMC."
        )
    )
    parser.add_argument("--code", type=Path, default=DEFAULT_CODE)
    parser.add_argument("--component", choices=["x", "z"], default="x")
    parser.add_argument("--points", type=int, default=51)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--approx-samples", type=int, default=10000)
    parser.add_argument("--mcmc-burnin", type=int, default=2000)
    parser.add_argument("--mcmc-thin", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--order",
        choices=["weight", "stored"],
        default="weight",
        help="Logical chain-rule order. 'weight' sorts stored logical rows by physical weight.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def code_key(name: str, n: int) -> str:
    base = name.replace("_HxHzLxLz", "")
    if base in {"gross", "two_gross"}:
        return f"{base}{n}"
    return re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_")


def component_matrices_and_logicals(code, component: str) -> tuple[np.ndarray, np.ndarray]:
    if component == "x":
        if code.lz is None:
            raise ValueError("X-component logical decomposition requires Lz rows")
        return code.hz, code.lz
    if code.lx is None:
        raise ValueError("Z-component logical decomposition requires Lx rows")
    return code.hx, code.lx


def sample_error(rng: np.random.Generator, n: int, p: float) -> np.ndarray:
    return (rng.random(n) < p).astype(np.uint8)


def logical_label(value: int, logical_rows: tuple[int, ...]) -> int:
    label = 0
    for idx, row in enumerate(logical_rows):
        if (value & row).bit_count() & 1:
            label |= 1 << idx
    return label


def project_label(label: int, order: list[int], width: int) -> int:
    projected = 0
    for out_idx, logical_idx in enumerate(order[:width]):
        if (label >> logical_idx) & 1:
            projected |= 1 << out_idx
    return projected


def prefix_entropy(label_logs: dict[int, float], order: list[int], width: int) -> float:
    if width == 0:
        return 0.0
    grouped: dict[int, float] = {}
    for label, log_weight in label_logs.items():
        logaddexp_dict(grouped, project_label(label, order, width), log_weight)
    return entropy_from_log_weights(list(grouped.values()))


def chain_contributions(label_logs: dict[int, float], order: list[int]) -> list[float]:
    entropies = [0.0]
    entropies.extend(prefix_entropy(label_logs, order, width) for width in range(1, len(order) + 1))
    return [entropies[idx + 1] - entropies[idx] for idx in range(len(order))]


def mcmc_label_log_weights(
    *,
    checks: np.ndarray,
    logical_rows: tuple[int, ...],
    syndrome: np.ndarray,
    p_one: np.ndarray,
    rng: np.random.Generator,
    samples: int,
    burnin: int,
    thin: int,
) -> tuple[dict[int, float], int, int, float]:
    particular, basis = affine_solution_space(checks, syndrome, p_one)
    weight_model = LogWeightModel.from_probabilities(p_one)

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

    label_logs: dict[int, float] = {}
    for value in unique:
        log_weight = weight_model.log_weight(value)
        if math.isfinite(log_weight):
            logaddexp_dict(label_logs, logical_label(value, logical_rows), log_weight)

    acceptance = accepted / proposals if proposals else 0.0
    return label_logs, len(basis), len(unique), acceptance


def point_estimate(
    *,
    code,
    component: str,
    p: float,
    order: list[int],
    runs: int,
    rng: np.random.Generator,
    approx_samples: int,
    mcmc_burnin: int,
    mcmc_thin: int,
) -> dict[str, float | list[float]]:
    checks, logical_matrix = component_matrices_and_logicals(code, component)
    k = int(logical_matrix.shape[0])
    if p == 0.0:
        return {
            "class_entropy": 0.0,
            "class_entropy_stderr": 0.0,
            "chain": [0.0] * k,
            "chain_stderr": [0.0] * k,
            "mean_affine_dimension": 0.0,
            "mean_list_size": 1.0,
            "mean_acceptance_rate": 0.0,
        }
    if p == 0.5:
        affine_dim = float(code.n - (code.rank_hz if component == "x" else code.rank_hx))
        return {
            "class_entropy": float(k),
            "class_entropy_stderr": 0.0,
            "chain": [1.0] * k,
            "chain_stderr": [0.0] * k,
            "mean_affine_dimension": affine_dim,
            "mean_list_size": float(2**min(k, 20)),
            "mean_acceptance_rate": 1.0,
        }

    logical_rows = int_rows_from_matrix(logical_matrix)
    p_one = np.full(code.n, p, dtype=float)
    class_values = np.empty(runs, dtype=float)
    chain_values = np.empty((runs, k), dtype=float)
    affine_dims = np.empty(runs, dtype=float)
    list_sizes = np.empty(runs, dtype=float)
    acceptances = np.empty(runs, dtype=float)
    for run_idx in range(runs):
        error = sample_error(rng, code.n, p)
        syndrome = (checks @ error % 2).astype(np.uint8)
        label_logs, affine_dim, list_size, acceptance = mcmc_label_log_weights(
            checks=checks,
            logical_rows=logical_rows,
            syndrome=syndrome,
            p_one=p_one,
            rng=rng,
            samples=approx_samples,
            burnin=mcmc_burnin,
            thin=mcmc_thin,
        )
        class_values[run_idx] = entropy_from_log_weights(list(label_logs.values()))
        chain_values[run_idx] = np.asarray(chain_contributions(label_logs, order), dtype=float)
        affine_dims[run_idx] = affine_dim
        list_sizes[run_idx] = list_size
        acceptances[run_idx] = acceptance

    def stderr(values: np.ndarray) -> float:
        return float(values.std(ddof=1) / math.sqrt(runs)) if runs > 1 else 0.0

    return {
        "class_entropy": float(class_values.mean()),
        "class_entropy_stderr": stderr(class_values),
        "chain": [float(value) for value in chain_values.mean(axis=0)],
        "chain_stderr": [stderr(chain_values[:, idx]) for idx in range(k)],
        "mean_affine_dimension": float(affine_dims.mean()),
        "mean_list_size": float(list_sizes.mean()),
        "mean_acceptance_rate": float(acceptances.mean()),
    }


def smooth_derivative(values: np.ndarray) -> np.ndarray:
    if values.size < 5:
        return values
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
    kernel /= kernel.sum()
    padded = np.pad(values, (2, 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def compute_result(args: argparse.Namespace) -> dict:
    code = load_code(args.code.resolve())
    _checks, logical_matrix = component_matrices_and_logicals(code, args.component)
    logical_weights = [int(weight) for weight in np.sum(logical_matrix % 2, axis=1)]
    if args.order == "weight":
        order = sorted(range(len(logical_weights)), key=lambda idx: (logical_weights[idx], idx))
    else:
        order = list(range(len(logical_weights)))

    rng = np.random.default_rng(args.seed)
    p_grid = np.linspace(0.0, 0.5, args.points)
    rows: list[dict[str, float]] = []
    for p in p_grid:
        estimate = point_estimate(
            code=code,
            component=args.component,
            p=float(p),
            order=order,
            runs=args.runs,
            rng=rng,
            approx_samples=args.approx_samples,
            mcmc_burnin=args.mcmc_burnin,
            mcmc_thin=args.mcmc_thin,
        )
        row: dict[str, float] = {
            "p": float(p),
            f"posterior_{args.component}_class_mcmc": float(estimate["class_entropy"]),
            f"posterior_{args.component}_class_mcmc_stderr": float(
                estimate["class_entropy_stderr"]
            ),
            f"posterior_{args.component}_class_component_norm_mcmc": float(
                estimate["class_entropy"]
            )
            / code.n,
            "mean_affine_dimension": float(estimate["mean_affine_dimension"]),
            "mean_list_size": float(estimate["mean_list_size"]),
            "mean_acceptance_rate": float(estimate["mean_acceptance_rate"]),
        }
        chain = estimate["chain"]
        chain_stderr = estimate["chain_stderr"]
        assert isinstance(chain, list)
        assert isinstance(chain_stderr, list)
        for pos, logical_idx in enumerate(order):
            prefix = f"logical_{logical_idx:02d}_chain"
            row[f"{prefix}_entropy_mcmc"] = float(chain[pos])
            row[f"{prefix}_entropy_mcmc_stderr"] = float(chain_stderr[pos])
            row[f"{prefix}_component_norm_mcmc"] = float(chain[pos]) / code.n
        rows.append(row)

    chain_norm_keys = [
        f"logical_{logical_idx:02d}_chain_component_norm_mcmc" for logical_idx in order
    ]
    p_values = np.asarray([row["p"] for row in rows], dtype=float)
    component_derivatives = []
    for key in chain_norm_keys:
        y = np.asarray([row[key] for row in rows], dtype=float)
        dy = smooth_derivative(np.gradient(y, p_values))
        dy[0] = 0.0
        dy[-1] = 0.0
        component_derivatives.append(dy)
    total_derivative = np.sum(np.vstack(component_derivatives), axis=0)
    peak_idx = int(np.argmax(total_derivative))
    peak_value = float(total_derivative[peak_idx])
    scale = 1.0 / peak_value if peak_value > 0.0 else 1.0

    for row_idx, row in enumerate(rows):
        row[f"posterior_{args.component}_class_component_norm_dp_decomp_mcmc"] = float(
            total_derivative[row_idx]
        )
        row[f"scaled_posterior_{args.component}_class_component_norm_dp_decomp_mcmc"] = float(
            total_derivative[row_idx] * scale
        )
        for key, dy in zip(chain_norm_keys, component_derivatives):
            dp_key = key.replace("_component_norm_mcmc", "_component_norm_dp_mcmc")
            row[dp_key] = float(dy[row_idx])
            row[f"scaled_{dp_key}"] = float(dy[row_idx] * scale)

    return {
        "code": {
            "path": str(args.code.resolve()),
            "name": code.name,
            "n": code.n,
            "k": code.k,
            "rank_hx": code.rank_hx,
            "rank_hz": code.rank_hz,
        },
        "component": args.component,
        "method": "posterior MCMC logical chain-rule decomposition",
        "runs": args.runs,
        "approx_samples": args.approx_samples,
        "mcmc_burnin": args.mcmc_burnin,
        "mcmc_thin": args.mcmc_thin,
        "seed": args.seed,
        "logical_order_kind": args.order,
        "logical_order": order,
        "logical_weights": logical_weights,
        "scaling": {
            "peak_p": float(rows[peak_idx]["p"]),
            "peak_gexit": peak_value,
            "scale_to_unit_peak": scale,
            "target_area_k_over_n": code.k / code.n,
        },
        "notes": {
            "decomposition": (
                "For a fixed logical basis/order, H(L|S) is decomposed by "
                "the chain rule into H(L_i | S, previous logical bits). The "
                "component derivatives sum to the total decomposition derivative."
            ),
            "basis_dependence": (
                "The decomposition is meaningful as a diagnostic, but not "
                "basis invariant. Compare stored-order, weight-order, and "
                "multiple seeds before treating clusters as structural."
            ),
        },
        "points": rows,
    }


def write_csv(result: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(result["points"][0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["points"])


def plot_decomposition(result: dict, path: Path) -> None:
    points = result["points"]
    p = np.asarray([row["p"] for row in points], dtype=float)
    component = result["component"]
    total_key = f"scaled_posterior_{component}_class_component_norm_dp_decomp_mcmc"
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(p, [row[total_key] for row in points], color="black", linewidth=1.7, label="sum")
    colors = plt.get_cmap("tab20").colors
    for pos, logical_idx in enumerate(result["logical_order"]):
        key = f"scaled_logical_{logical_idx:02d}_chain_component_norm_dp_mcmc"
        weight = result["logical_weights"][logical_idx]
        ax.plot(
            p,
            [row[key] for row in points],
            linewidth=0.9,
            color=colors[pos % len(colors)],
            label=f"L{logical_idx} w={weight}",
        )
    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlabel(r"BSC crossover probability $p$")
    ax.set_ylabel(r"scaled derivative contribution")
    ax.set_title("Logical chain-rule BSC decomposition")
    ax.set_xlim(0.0, 0.5)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    result = compute_result(args)
    key = code_key(result["code"]["name"], result["code"]["n"])
    out_dir = args.out_dir.resolve()
    stem = f"{key}_bsc_{args.component}_logical_decomp_mcmc_{args.order}"
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"
    png_path = out_dir / f"{stem}.png"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(result, csv_path)
    plot_decomposition(result, png_path)
    for path in (json_path, csv_path, png_path):
        print(f"wrote {path}")
    print(
        "decomposition peak: "
        f"p={result['scaling']['peak_p']:.4g}, "
        f"g={result['scaling']['peak_gexit']:.8g}, "
        f"target area k/n={result['scaling']['target_area_k_over_n']:.8g}"
    )


if __name__ == "__main__":
    main()
