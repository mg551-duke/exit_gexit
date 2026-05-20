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

from experiments.beyond_erasure_diagnostics import posterior_entropy_for_sample
from experiments.exit_curve_experiments import ROOT, load_code


DEFAULT_CODE = ROOT / "codes" / "gross_HxHzLxLz.npz"
DEFAULT_OUT_DIR = ROOT / "data" / "experiments" / "gexit_curves"
DEFAULT_TIKZ_DIR = DEFAULT_OUT_DIR / "tikz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample BSC correction-class entropy curves using posterior MCMC. "
            "This is intended for codes where exact GEXIT contraction is too wide."
        )
    )
    parser.add_argument("--code", type=Path, default=DEFAULT_CODE)
    parser.add_argument("--component", choices=["x", "z"], default="x")
    parser.add_argument("--points", type=int, default=26)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--approx-samples", type=int, default=1500)
    parser.add_argument("--mcmc-burnin", type=int, default=1000)
    parser.add_argument("--mcmc-thin", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tikz-dir", type=Path, default=DEFAULT_TIKZ_DIR)
    return parser.parse_args()


def code_key(name: str, n: int) -> str:
    base = name.replace("_HxHzLxLz", "")
    if base in {"gross", "two_gross"}:
        return f"{base}{n}"
    return re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_")


def axis_key(key: str) -> str:
    parts = re.split(r"[^A-Za-z0-9]+", key)
    return "".join(part[:1].upper() + part[1:] for part in parts if part)


def tex_code_label(name: str, n: int, k: int) -> str:
    base = name.replace("_HxHzLxLz", "").replace("_", r"\_")
    return rf"{base} $[[{n},{k}]]$"


def format_scale(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) >= 1000.0 or abs(value) < 0.01:
        exponent = int(math.floor(math.log10(abs(value))))
        mantissa = value / (10.0**exponent)
        return rf"{mantissa:.3g}\times 10^{{{exponent}}}"
    return f"{value:.3g}"


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


def transformed_derivative_values(result: dict) -> tuple[np.ndarray, np.ndarray, float]:
    points = result["points"]
    derivative_key = result["keys"]["derivative"]
    p = np.array([point["p"] for point in points], dtype=float)
    dydp = np.array([point[derivative_key] for point in points], dtype=float)
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


def sample_error(rng: np.random.Generator, n: int, p: float) -> np.ndarray:
    return (rng.random(n) < p).astype(np.uint8)


def component_matrices(code, component: str) -> tuple[np.ndarray, np.ndarray]:
    if component == "x":
        return code.hz, code.hx
    return code.hx, code.hz


def estimate_point(
    *,
    code,
    component: str,
    p: float,
    runs: int,
    rng: np.random.Generator,
    approx_samples: int,
    mcmc_burnin: int,
    mcmc_thin: int,
) -> dict[str, float]:
    if p == 0.0:
        return {
            "posterior_error": 0.0,
            "posterior_error_stderr": 0.0,
            "posterior_class": 0.0,
            "posterior_class_stderr": 0.0,
            "posterior_degeneracy_gain": 0.0,
            "posterior_degeneracy_gain_stderr": 0.0,
            "mean_affine_dimension": 0.0,
            "mean_list_size": 1.0,
        }
    if p == 0.5:
        affine_dim = float(
            code.n - (code.rank_hz if component == "x" else code.rank_hx)
        )
        class_entropy = float(code.k)
        return {
            "posterior_error": affine_dim,
            "posterior_error_stderr": 0.0,
            "posterior_class": class_entropy,
            "posterior_class_stderr": 0.0,
            "posterior_degeneracy_gain": affine_dim - class_entropy,
            "posterior_degeneracy_gain_stderr": 0.0,
            "mean_affine_dimension": affine_dim,
            "mean_list_size": float(2**min(code.k, 20)),
        }

    checks, quotient_rows = component_matrices(code, component)
    p_one = np.full(code.n, p, dtype=float)
    error_values = np.empty(runs, dtype=float)
    class_values = np.empty(runs, dtype=float)
    gain_values = np.empty(runs, dtype=float)
    affine_dims = np.empty(runs, dtype=float)
    list_sizes = np.empty(runs, dtype=float)
    for run_idx in range(runs):
        error = sample_error(rng, code.n, p)
        syndrome = (checks @ error % 2).astype(np.uint8)
        result = posterior_entropy_for_sample(
            checks,
            quotient_rows,
            syndrome,
            p_one,
            rng,
            max_exact_affine_dim=-1,
            approx_samples=approx_samples,
            mcmc_burnin=mcmc_burnin,
            mcmc_thin=mcmc_thin,
        )
        error_values[run_idx] = result.error_entropy
        class_values[run_idx] = result.class_entropy
        gain_values[run_idx] = result.degeneracy_gain
        affine_dims[run_idx] = result.affine_dimension
        list_sizes[run_idx] = result.list_size

    def sample_stderr(values: np.ndarray) -> float:
        return float(values.std(ddof=1) / math.sqrt(runs)) if runs > 1 else 0.0

    return {
        "posterior_error": float(error_values.mean()),
        "posterior_error_stderr": sample_stderr(error_values),
        "posterior_class": float(class_values.mean()),
        "posterior_class_stderr": sample_stderr(class_values),
        "posterior_degeneracy_gain": float(gain_values.mean()),
        "posterior_degeneracy_gain_stderr": sample_stderr(gain_values),
        "mean_affine_dimension": float(affine_dims.mean()),
        "mean_list_size": float(list_sizes.mean()),
    }


def isotonic_nondecreasing(values: np.ndarray) -> np.ndarray:
    """Least-squares nondecreasing projection by the pool-adjacent-violators algorithm."""
    levels: list[float] = []
    weights: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    for idx, value in enumerate(values):
        levels.append(float(value))
        weights.append(1.0)
        starts.append(idx)
        ends.append(idx)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            total_weight = weights[-2] + weights[-1]
            merged = (levels[-2] * weights[-2] + levels[-1] * weights[-1]) / total_weight
            levels[-2] = merged
            weights[-2] = total_weight
            ends[-2] = ends[-1]
            levels.pop()
            weights.pop()
            starts.pop()
            ends.pop()

    fitted = np.empty_like(values, dtype=float)
    for level, start, end in zip(levels, starts, ends):
        fitted[start : end + 1] = level
    return fitted


def smooth_derivative(values: np.ndarray) -> np.ndarray:
    if values.size < 5:
        return values
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
    kernel /= kernel.sum()
    padded = np.pad(values, (2, 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def compute_result(args: argparse.Namespace) -> dict:
    code = load_code(args.code.resolve())
    rng = np.random.default_rng(args.seed)
    p_grid = np.linspace(0.0, 0.5, args.points)
    rows = []
    for p in p_grid:
        point = estimate_point(
            code=code,
            component=args.component,
            p=float(p),
            runs=args.runs,
            rng=rng,
            approx_samples=args.approx_samples,
            mcmc_burnin=args.mcmc_burnin,
            mcmc_thin=args.mcmc_thin,
        )
        class_entropy = point["posterior_class"]
        rows.append(
            {
                "p": float(p),
                f"posterior_{args.component}_error_mcmc": point["posterior_error"],
                f"posterior_{args.component}_error_mcmc_stderr": point[
                    "posterior_error_stderr"
                ],
                f"posterior_{args.component}_error_component_norm_mcmc": point[
                    "posterior_error"
                ]
                / code.n,
                f"posterior_{args.component}_class_mcmc": class_entropy,
                f"posterior_{args.component}_class_mcmc_stderr": point[
                    "posterior_class_stderr"
                ],
                f"posterior_{args.component}_class_component_norm_mcmc": class_entropy
                / code.n,
                f"posterior_{args.component}_degeneracy_gain_mcmc": point[
                    "posterior_degeneracy_gain"
                ],
                f"posterior_{args.component}_degeneracy_gain_mcmc_stderr": point[
                    "posterior_degeneracy_gain_stderr"
                ],
                f"posterior_{args.component}_degeneracy_gain_component_norm_mcmc": point[
                    "posterior_degeneracy_gain"
                ]
                / code.n,
                "mean_affine_dimension": point["mean_affine_dimension"],
                "mean_list_size": point["mean_list_size"],
            }
        )

    y_key = f"posterior_{args.component}_class_component_norm_mcmc"
    smooth_y_key = f"posterior_{args.component}_class_component_norm_mcmc_isotonic"
    dp_key = f"posterior_{args.component}_class_component_norm_dp_mcmc"
    scaled_key = f"scaled_{dp_key}"
    y = np.array([row[y_key] for row in rows], dtype=float)
    y_isotonic = isotonic_nondecreasing(y)
    y_isotonic[0] = y[0]
    y_isotonic[-1] = y[-1]
    dydp = smooth_derivative(np.gradient(y_isotonic, p_grid))
    dydp[0] = 0.0
    dydp[-1] = 0.0
    dydp = np.maximum(dydp, 0.0)
    peak_idx = int(np.argmax(dydp))
    peak_value = float(dydp[peak_idx])
    scale = 1.0 / peak_value if peak_value > 0.0 else 1.0
    for row, smooth_value, value in zip(rows, y_isotonic, dydp):
        row[smooth_y_key] = float(smooth_value)
        row[dp_key] = float(value)
        row[scaled_key] = float(value * scale)

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
        "method": "posterior MCMC class-entropy estimate",
        "runs": args.runs,
        "approx_samples": args.approx_samples,
        "mcmc_burnin": args.mcmc_burnin,
        "mcmc_thin": args.mcmc_thin,
        "seed": args.seed,
        "keys": {
            "error": f"posterior_{args.component}_error_mcmc",
            "error_norm": f"posterior_{args.component}_error_component_norm_mcmc",
            "class": f"posterior_{args.component}_class_mcmc",
            "class_norm": y_key,
            "class_norm_isotonic": smooth_y_key,
            "degeneracy_gain": f"posterior_{args.component}_degeneracy_gain_mcmc",
            "degeneracy_gain_norm": f"posterior_{args.component}_degeneracy_gain_component_norm_mcmc",
            "derivative": dp_key,
            "scaled_derivative": scaled_key,
        },
        "notes": {
            "error_entropy": (
                "The MCMC error entropy is computed on the unique visited "
                "posterior list and can underestimate H(x|S) when many "
                "physical errors contribute within a correction class."
            ),
            "class_entropy": (
                "The class entropy is computed by summing the same visited "
                "error weights by correction class modulo the stabilizer "
                "rowspace."
            ),
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
    component = result["component"]
    key = code_key(code["name"], code["n"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tikz_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{key}_bsc_{component}_class_mcmc.json"
    csv_path = out_dir / f"{key}_bsc_{component}_class_mcmc.csv"
    bit_png = out_dir / f"{key}_bsc_{component}_bit_scale_mcmc.png"
    bit_entropy_png = out_dir / f"{key}_bsc_{component}_bit_scale_entropy_axis_mcmc.png"
    class_png = out_dir / f"{key}_bsc_{component}_class_entropy_mcmc.png"
    derivative_png = out_dir / f"{key}_scaled_bsc_{component}_gexit_mcmc.png"
    derivative_entropy_png = out_dir / f"{key}_scaled_bsc_{component}_gexit_entropy_axis_mcmc.png"
    class_tikz = tikz_dir / f"fig_{key}_bsc_{component}_class_entropy_mcmc_compact.tex"
    derivative_tikz = tikz_dir / f"fig_scaled_bsc_{component}_gexit_{key}_mcmc_compact.tex"

    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    fields = list(result["points"][0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result["points"])

    plot_bit_scale(result, bit_png)
    plot_bit_scale_entropy_axis(result, bit_entropy_png)
    plot_class_entropy(result, class_png)
    plot_scaled_derivative(result, derivative_png)
    plot_scaled_derivative_entropy_axis(result, derivative_entropy_png)
    write_class_tikz(result, class_tikz)
    write_derivative_tikz(result, derivative_tikz)

    for path in (
        json_path,
        csv_path,
        bit_png,
        bit_entropy_png,
        class_png,
        derivative_png,
        derivative_entropy_png,
        class_tikz,
        derivative_tikz,
    ):
        print(f"wrote {path}")


def plot_bit_scale(result: dict, out_path: Path) -> None:
    code = result["code"]
    points = result["points"]
    keys = result["keys"]
    p = np.array([point["p"] for point in points], dtype=float)
    series = [
        (keys["error"], r"$\mathbb{E}H(x\mid S)$"),
        (keys["class"], r"$\mathbb{E}H(C_X\mid S)$"),
        (keys["degeneracy_gain"], r"$\mathbb{E}H(x\mid C_X,S)$"),
    ]

    fig, ax = plt.subplots(figsize=(6.1, 4.1))
    for key, label in series:
        ax.plot(
            p,
            [point[key] for point in points],
            marker=".",
            linewidth=1.45,
            label=label,
        )
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
    code = result["code"]
    points = result["points"]
    keys = result["keys"]
    component = result["component"]
    component_label = component.upper()
    p = np.array([point["p"] for point in points], dtype=float)
    t = binary_entropy_axis(p)
    series = [
        (keys["error"], rf"$\mathbb{{E}}H({component}\mid S)$"),
        (keys["class"], rf"$\mathbb{{E}}H(C_{component_label}\mid S)$"),
        (
            keys["degeneracy_gain"],
            rf"$\mathbb{{E}}H({component}\mid C_{component_label},S)$",
        ),
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


def plot_class_entropy(result: dict, out_path: Path) -> None:
    code = result["code"]
    points = result["points"]
    class_key = result["keys"]["class"]
    stderr_key = f"{class_key}_stderr"
    p = np.array([point["p"] for point in points], dtype=float)
    y = np.array([point[class_key] for point in points], dtype=float)
    yerr = np.array([point[stderr_key] for point in points], dtype=float)

    fig, ax = plt.subplots(figsize=(6.1, 4.1))
    ax.axhline(code["k"], color="black", linestyle="--", linewidth=0.9, label=r"$k$")
    ax.errorbar(p, y, yerr=yerr, marker=".", linewidth=1.45, capsize=2.0, label=r"$\mathbb{E}H(C_X\mid S)$")
    ax.set_xlabel(r"BSC crossover probability $p$")
    ax.set_ylabel("expected class entropy (bits)")
    title_label = code["name"].replace("_HxHzLxLz", "").replace("_", " ")
    ax.set_title(rf"{title_label}: sampled BSC correction class")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_scaled_derivative(result: dict, out_path: Path) -> None:
    code = result["code"]
    points = result["points"]
    scaled_key = result["keys"]["scaled_derivative"]
    scale = result["scaling"]["scale_to_unit_peak"]
    p = np.array([point["p"] for point in points], dtype=float)
    y = np.array([point[scaled_key] for point in points], dtype=float)

    fig, ax = plt.subplots(figsize=(6.1, 4.1))
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
    ax.axvline(0.5, color="black", linestyle=":", linewidth=0.9)
    ax.plot(
        p,
        y,
        marker=".",
        linewidth=1.45,
        label=rf"{tex_code_label(code['name'], code['n'], code['k'])}, scaled by ${format_scale(scale)}$",
    )
    ax.set_xlabel(r"BSC crossover probability $p$")
    ax.set_ylabel(r"$\widetilde g_X^{\rm BSC}(p)$")
    ax.set_title("Sampled BSC correction-class derivative")
    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(-0.02, 1.06)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_scaled_derivative_entropy_axis(result: dict, out_path: Path) -> None:
    code = result["code"]
    component_label = result["component"].upper()
    t, dydt, peak = transformed_derivative_values(result)
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
        label=rf"{tex_code_label(code['name'], code['n'], code['k'])}, scaled by ${format_scale(scale)}$",
    )
    ax.set_xlabel(r"BSC channel entropy $t=h_2(p)$")
    ax.set_ylabel(rf"$\widetilde g_{component_label}^{{\rm BSC}}(t)$")
    ax.set_title("Sampled BSC correction-class derivative")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.02, 1.06)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_class_tikz(result: dict, out_path: Path) -> None:
    code = result["code"]
    key = code_key(code["name"], code["n"])
    axis = axis_key(key)
    class_key = result["keys"]["class"]
    ymax = max(code["k"] * 1.05, max(point[class_key] for point in result["points"]) * 1.05)
    lines = [
        f"% Compact PGFPlots panel for sampled {key} BSC correction-class entropy.",
        "% Requires \\usepackage{pgfplots} and \\pgfplotsset{compat=1.18}.",
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        f"  name={axis}BSCClassEntropyAxis,",
        r"  width=\linewidth,",
        r"  height=0.78\linewidth,",
        r"  title={BSC correction-class entropy},",
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
        r"\addplot+[color=orange, mark=square*, mark size=1.0pt, line width=0.65pt]",
        rf"coordinates {{{tikz_coordinates(result['points'], class_key)}}};",
        r"\addlegendentry{$\mathbb{E}H(C_X\mid S)$}",
        "\\end{axis}",
        "\\end{tikzpicture}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_derivative_tikz(result: dict, out_path: Path) -> None:
    code = result["code"]
    key = code_key(code["name"], code["n"])
    axis = axis_key(key)
    scaled_key = result["keys"]["scaled_derivative"]
    scale = result["scaling"]["scale_to_unit_peak"]
    lines = [
        f"% Compact PGFPlots panel for sampled {key} BSC correction-class derivative.",
        "% Requires \\usepackage{pgfplots} and \\pgfplotsset{compat=1.18}.",
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        f"  name=scaledBSCGEXIT{axis}Axis,",
        r"  width=\linewidth,",
        r"  height=0.78\linewidth,",
        r"  title={BSC correction-class derivative},",
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
        rf"coordinates {{{tikz_coordinates(result['points'], scaled_key)}}};",
        rf"\addlegendentry{{{tex_code_label(code['name'], code['n'], code['k'])}, scaled by ${format_scale(scale)}$}}",
        "\\end{axis}",
        "\\end{tikzpicture}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    result = compute_result(args)
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
