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
    entropy_from_log_weights,
    logaddexp_dict,
)
from experiments.bsc_gexit_logical_decomposition_mcmc import (
    code_key,
    component_matrices_and_logicals,
    mcmc_label_log_weights,
    sample_error,
    smooth_derivative,
)
from experiments.exit_curve_experiments import ROOT, load_code


DEFAULT_CODE = ROOT / "codes" / "gross_HxHzLxLz.npz"
DEFAULT_OUT_DIR = (
    ROOT
    / "data"
    / "experiments"
    / "gexit_curves"
    / "gross144_logical_decomp_separate_weight"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate per-logical BSC posterior diagnostics H(L_i|S) and "
            "H(L_i|S,L_others) using the same MCMC label posterior as the "
            "logical chain-rule decomposition."
        )
    )
    parser.add_argument("--code", type=Path, default=DEFAULT_CODE)
    parser.add_argument("--component", choices=["x", "z"], default="x")
    parser.add_argument("--points", type=int, default=51)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--approx-samples", type=int, default=10000)
    parser.add_argument("--mcmc-burnin", type=int, default=2000)
    parser.add_argument("--mcmc-thin", type=int, default=1)
    parser.add_argument("--seed", type=int, default=144)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def project_without_bit(label: int, bit_idx: int, k: int) -> int:
    projected = 0
    out_idx = 0
    for idx in range(k):
        if idx == bit_idx:
            continue
        if (label >> idx) & 1:
            projected |= 1 << out_idx
        out_idx += 1
    return projected


def logical_marginal_entropy(label_logs: dict[int, float], bit_idx: int) -> float:
    grouped: dict[int, float] = {}
    for label, log_weight in label_logs.items():
        logaddexp_dict(grouped, (label >> bit_idx) & 1, log_weight)
    return entropy_from_log_weights(list(grouped.values()))


def logical_conditional_given_others(
    label_logs: dict[int, float],
    bit_idx: int,
    k: int,
) -> float:
    total = entropy_from_log_weights(list(label_logs.values()))
    grouped: dict[int, float] = {}
    for label, log_weight in label_logs.items():
        logaddexp_dict(grouped, project_without_bit(label, bit_idx, k), log_weight)
    without_bit = entropy_from_log_weights(list(grouped.values()))
    return max(0.0, total - without_bit)


def point_estimate(
    *,
    code,
    component: str,
    p: float,
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
            "marginal": [0.0] * k,
            "marginal_stderr": [0.0] * k,
            "given_others": [0.0] * k,
            "given_others_stderr": [0.0] * k,
            "mean_affine_dimension": 0.0,
            "mean_list_size": 1.0,
            "mean_acceptance_rate": 0.0,
        }
    if p == 0.5:
        affine_dim = float(code.n - (code.rank_hz if component == "x" else code.rank_hx))
        return {
            "class_entropy": float(k),
            "class_entropy_stderr": 0.0,
            "marginal": [1.0] * k,
            "marginal_stderr": [0.0] * k,
            "given_others": [1.0] * k,
            "given_others_stderr": [0.0] * k,
            "mean_affine_dimension": affine_dim,
            "mean_list_size": float(2**min(k, 20)),
            "mean_acceptance_rate": 1.0,
        }

    logical_rows = tuple(
        int(sum(int(bit) << idx for idx, bit in enumerate(row % 2)))
        for row in logical_matrix.astype(np.uint8)
    )
    p_one = np.full(code.n, p, dtype=float)
    class_values = np.empty(runs, dtype=float)
    marginal_values = np.empty((runs, k), dtype=float)
    given_others_values = np.empty((runs, k), dtype=float)
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
        for logical_idx in range(k):
            marginal_values[run_idx, logical_idx] = logical_marginal_entropy(
                label_logs, logical_idx
            )
            given_others_values[run_idx, logical_idx] = (
                logical_conditional_given_others(label_logs, logical_idx, k)
            )
        affine_dims[run_idx] = affine_dim
        list_sizes[run_idx] = list_size
        acceptances[run_idx] = acceptance

    def stderr(values: np.ndarray) -> float:
        return float(values.std(ddof=1) / math.sqrt(runs)) if runs > 1 else 0.0

    return {
        "class_entropy": float(class_values.mean()),
        "class_entropy_stderr": stderr(class_values),
        "marginal": [float(value) for value in marginal_values.mean(axis=0)],
        "marginal_stderr": [stderr(marginal_values[:, idx]) for idx in range(k)],
        "given_others": [float(value) for value in given_others_values.mean(axis=0)],
        "given_others_stderr": [
            stderr(given_others_values[:, idx]) for idx in range(k)
        ],
        "mean_affine_dimension": float(affine_dims.mean()),
        "mean_list_size": float(list_sizes.mean()),
        "mean_acceptance_rate": float(acceptances.mean()),
    }


def compute_result(args: argparse.Namespace) -> dict:
    code = load_code(args.code.resolve())
    _checks, logical_matrix = component_matrices_and_logicals(code, args.component)
    k = int(logical_matrix.shape[0])
    logical_weights = [int(weight) for weight in np.sum(logical_matrix % 2, axis=1)]
    rng = np.random.default_rng(args.seed)
    p_grid = np.linspace(0.0, 0.5, args.points)
    rows: list[dict[str, float]] = []

    for p in p_grid:
        estimate = point_estimate(
            code=code,
            component=args.component,
            p=float(p),
            runs=args.runs,
            rng=rng,
            approx_samples=args.approx_samples,
            mcmc_burnin=args.mcmc_burnin,
            mcmc_thin=args.mcmc_thin,
        )
        row: dict[str, float] = {
            "p": float(p),
            f"posterior_{args.component}_class_mcmc": float(
                estimate["class_entropy"]
            ),
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
        for name in ("marginal", "given_others"):
            values = estimate[name]
            stderrs = estimate[f"{name}_stderr"]
            assert isinstance(values, list)
            assert isinstance(stderrs, list)
            for logical_idx in range(k):
                prefix = f"logical_{logical_idx:02d}_{name}"
                row[f"{prefix}_entropy_mcmc"] = float(values[logical_idx])
                row[f"{prefix}_entropy_mcmc_stderr"] = float(stderrs[logical_idx])
                row[f"{prefix}_component_norm_mcmc"] = (
                    float(values[logical_idx]) / code.n
                )
        rows.append(row)

    p_values = np.asarray([row["p"] for row in rows], dtype=float)
    class_norm_key = f"posterior_{args.component}_class_component_norm_mcmc"
    class_derivative = smooth_derivative(
        np.gradient(
            np.asarray([row[class_norm_key] for row in rows], dtype=float),
            p_values,
        )
    )
    class_derivative[0] = 0.0
    class_derivative[-1] = 0.0

    diagnostic_keys: dict[str, list[str]] = {}
    diagnostic_derivatives: dict[str, list[np.ndarray]] = {}
    for name in ("marginal", "given_others"):
        keys = [
            f"logical_{logical_idx:02d}_{name}_component_norm_mcmc"
            for logical_idx in range(k)
        ]
        diagnostic_keys[name] = keys
        diagnostic_derivatives[name] = []
        for key in keys:
            dy = smooth_derivative(
                np.gradient(
                    np.asarray([row[key] for row in rows], dtype=float),
                    p_values,
                )
            )
            dy[0] = 0.0
            dy[-1] = 0.0
            diagnostic_derivatives[name].append(dy)

    peak_value = float(np.max(class_derivative))
    scale = 1.0 / peak_value if peak_value > 0.0 else 1.0
    for row_idx, row in enumerate(rows):
        row[f"posterior_{args.component}_class_component_norm_dp_mcmc"] = float(
            class_derivative[row_idx]
        )
        row[f"scaled_posterior_{args.component}_class_component_norm_dp_mcmc"] = float(
            class_derivative[row_idx] * scale
        )
        for name, keys in diagnostic_keys.items():
            total = sum(dy[row_idx] for dy in diagnostic_derivatives[name])
            row[f"{name}_sum_component_norm_dp_mcmc"] = float(total)
            row[f"scaled_{name}_sum_component_norm_dp_mcmc"] = float(total * scale)
            for key, dy in zip(keys, diagnostic_derivatives[name]):
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
        "method": "posterior MCMC logical marginal and leave-one-out conditionals",
        "runs": args.runs,
        "approx_samples": args.approx_samples,
        "mcmc_burnin": args.mcmc_burnin,
        "mcmc_thin": args.mcmc_thin,
        "seed": args.seed,
        "logical_weights": logical_weights,
        "scaling": {
            "scale_to_class_derivative_peak": scale,
            "class_derivative_peak": peak_value,
            "target_area_k_over_n": code.k / code.n,
        },
        "notes": {
            "marginal": "logical_i_marginal_entropy_mcmc estimates E_S H(L_i | S).",
            "given_others": (
                "logical_i_given_others_entropy_mcmc estimates "
                "E_S H(L_i | S, L_{j != i})."
            ),
            "sums": (
                "The per-logical sums are diagnostic totals and are not equal "
                "to H(L|S) in general."
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


def plot_separate(result: dict, out_dir: Path, diagnostic: str) -> None:
    points = result["points"]
    component = result["component"]
    p = np.asarray([row["p"] for row in points], dtype=float)
    class_key = f"scaled_posterior_{component}_class_component_norm_dp_mcmc"
    sum_key = f"scaled_{diagnostic}_sum_component_norm_dp_mcmc"
    class_y = np.asarray([row[class_key] for row in points], dtype=float)
    sum_y = np.asarray([row[sum_key] for row in points], dtype=float)
    show_diagnostic_sum = diagnostic != "given_others"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = {
        "marginal": r"$H(L_i\mid S)$",
        "given_others": r"$H(L_i\mid S,L_{\mathrm{others}})$",
    }
    titles = {
        "marginal": "marginal logical uncertainty",
        "given_others": "leave-one-out logical uncertainty",
    }
    k = int(result["code"]["k"])
    for logical_idx in range(k):
        key = f"scaled_logical_{logical_idx:02d}_{diagnostic}_component_norm_dp_mcmc"
        y = np.asarray([row[key] for row in points], dtype=float)
        weight = result["logical_weights"][logical_idx]

        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.45)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.plot(p, class_y, color="black", linewidth=1.8, label=r"$dH(L|S)$")
        if show_diagnostic_sum:
            ax.plot(
                p,
                sum_y,
                color="0.45",
                linestyle=":",
                linewidth=1.4,
                label=f"sum of {diagnostic} terms",
            )
        ax.plot(
            p,
            y,
            color="#1f77b4",
            linewidth=1.5,
            marker=".",
            markersize=3,
            label=f"L{logical_idx:02d}: {labels[diagnostic]}",
        )
        ax.set_xlabel(r"BSC crossover probability $p$")
        ax.set_ylabel("scaled derivative contribution")
        ax.set_title(
            f"Gross [[144,12]] L{logical_idx:02d} {titles[diagnostic]} "
            f"(weight {weight})"
        )
        ax.set_xlim(0.0, 0.5)
        plotted = [y, class_y]
        if show_diagnostic_sum:
            plotted.append(sum_y)
        ymin = min(*(float(values.min()) for values in plotted), -0.05)
        ymax = max(*(float(values.max()) for values in plotted), 1.05)
        pad = 0.06 * max(1.0, ymax - ymin)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / f"gross144_bsc_x_logical_{logical_idx:02d}_{diagnostic}_with_sum.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    result = compute_result(args)
    key = code_key(result["code"]["name"], result["code"]["n"])
    out_dir = args.out_dir.resolve()
    stem = f"{key}_bsc_{args.component}_logical_bit_conditionals_mcmc"
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(result, csv_path)
    plot_separate(result, out_dir / "marginal_given_s", "marginal")
    plot_separate(result, out_dir / "given_s_and_other_logicals", "given_others")
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote plots under {out_dir / 'marginal_given_s'}")
    print(f"wrote plots under {out_dir / 'given_s_and_other_logicals'}")


if __name__ == "__main__":
    main()
