from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exit_curve_experiments import ROOT, load_code


DEFAULT_CODE = ROOT / "codes" / "surface5_HxHzLxLz.npz"
DEFAULT_OUT_DIR = ROOT / "data" / "experiments" / "gexit_curves"
DEFAULT_TIKZ_DIR = DEFAULT_OUT_DIR / "tikz"


@dataclass(frozen=True)
class ObservationBasis:
    syndrome_rank: int
    quotient_rank: int
    rows: tuple[int, ...]
    column_masks: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exact BSC/GEXIT entropy curves for the X component of the d=5 "
            "surface code."
        )
    )
    parser.add_argument("--code", type=Path, default=DEFAULT_CODE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tikz-dir", type=Path, default=DEFAULT_TIKZ_DIR)
    parser.add_argument("--points", type=int, default=101)
    parser.add_argument("--plot", action="store_true")
    return parser.parse_args()


def row_to_int(row: np.ndarray) -> int:
    value = 0
    for col in np.flatnonzero(row % 2):
        value |= 1 << int(col)
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


def rank_int_rows(rows: Iterable[int]) -> int:
    basis: dict[int, int] = {}
    return sum(1 for row in rows if add_row_to_basis(row, basis))


def independent_rows(matrix: np.ndarray) -> tuple[int, ...]:
    basis: dict[int, int] = {}
    selected: list[int] = []
    for row in matrix:
        value = row_to_int(row)
        if add_row_to_basis(value, basis):
            selected.append(value)
    return tuple(selected)


def gf2_rref(matrix: np.ndarray) -> tuple[np.ndarray, list[int]]:
    a = (matrix.copy() % 2).astype(np.uint8)
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
        for other in range(m):
            if other != row and a[other, col]:
                a[other] ^= a[row]
        pivots.append(col)
        row += 1
        if row == m:
            break
    return a[:row], pivots


def nullspace_rows(matrix: np.ndarray) -> tuple[int, ...]:
    rref, pivots = gf2_rref(matrix)
    n = matrix.shape[1]
    pivot_set = set(pivots)
    rows: list[int] = []
    for free_col in range(n):
        if free_col in pivot_set:
            continue
        vector = np.zeros(n, dtype=np.uint8)
        vector[free_col] = 1
        for row_idx, pivot_col in enumerate(pivots):
            if rref[row_idx, free_col]:
                vector[pivot_col] = 1
        rows.append(row_to_int(vector))
    return tuple(rows)


def build_x_observation_basis(hx: np.ndarray, hz: np.ndarray) -> ObservationBasis:
    """Build a map whose kernel is the X-stabilizer rowspace.

    The first rows form an independent syndrome basis for H_Z.  The remaining
    rows extend that syndrome basis to all of ker(H_X), so the full observation
    records the pair (syndrome, X-correction class) up to an invertible relabeling.
    """

    syndrome_rows = list(independent_rows(hz))
    syndrome_rank = len(syndrome_rows)
    target_rank = hx.shape[1] - rank_int_rows(row_to_int(row) for row in hx)

    basis: dict[int, int] = {}
    full_rows: list[int] = []
    for row in syndrome_rows:
        if add_row_to_basis(row, basis):
            full_rows.append(row)

    for row in nullspace_rows(hx):
        if add_row_to_basis(row, basis):
            full_rows.append(row)
            if len(full_rows) == target_rank:
                break

    if len(full_rows) != target_rank:
        raise ValueError(
            f"could not extend H_Z to ker(H_X): got {len(full_rows)} rows, "
            f"expected {target_rank}"
        )

    n = hx.shape[1]
    column_masks: list[int] = []
    for col in range(n):
        mask = 0
        for row_idx, row in enumerate(full_rows):
            if (row >> col) & 1:
                mask |= 1 << row_idx
        column_masks.append(mask)

    return ObservationBasis(
        syndrome_rank=syndrome_rank,
        quotient_rank=target_rank,
        rows=tuple(full_rows),
        column_masks=tuple(column_masks),
    )


def binary_entropy(prob: float) -> float:
    if prob <= 0.0 or prob >= 1.0:
        return 0.0
    return -prob * math.log2(prob) - (1.0 - prob) * math.log2(1.0 - prob)


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
        [point["exact_x_class_component_norm_dp"] for point in points],
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


def xor_distribution_and_derivative(
    column_masks: tuple[int, ...],
    rank: int,
    p: float,
) -> tuple[np.ndarray, np.ndarray]:
    size = 1 << rank
    index = np.arange(size, dtype=np.uint32)
    dist = np.zeros(size, dtype=np.float64)
    deriv = np.zeros(size, dtype=np.float64)
    dist[0] = 1.0

    for mask, count in Counter(column_masks).items():
        one_minus_2p = 1.0 - 2.0 * p
        odd_prob = 0.5 * (1.0 - one_minus_2p**count)
        if count == 0:
            odd_deriv = 0.0
        else:
            odd_deriv = count * one_minus_2p ** (count - 1)

        shifted = dist[index ^ np.uint32(mask)]
        shifted_deriv = deriv[index ^ np.uint32(mask)]
        new_dist = (1.0 - odd_prob) * dist + odd_prob * shifted
        new_deriv = (
            (1.0 - odd_prob) * deriv
            + odd_prob * shifted_deriv
            + odd_deriv * (shifted - dist)
        )
        dist = new_dist
        deriv = new_deriv

    return dist, deriv


def entropy_and_derivative(dist: np.ndarray, deriv: np.ndarray) -> tuple[float, float]:
    mask = dist > 0.0
    if not np.any(mask):
        return 0.0, 0.0
    logp = np.log2(dist[mask])
    entropy = float(-(dist[mask] @ logp))
    derivative = float(-(deriv[mask] @ logp))
    return entropy, derivative


def marginalize_to_syndrome(
    dist: np.ndarray,
    deriv: np.ndarray,
    syndrome_rank: int,
    quotient_rank: int,
) -> tuple[np.ndarray, np.ndarray]:
    syndrome_size = 1 << syndrome_rank
    extra_size = 1 << (quotient_rank - syndrome_rank)
    reshaped_dist = dist.reshape((extra_size, syndrome_size))
    reshaped_deriv = deriv.reshape((extra_size, syndrome_size))
    return reshaped_dist.sum(axis=0), reshaped_deriv.sum(axis=0)


def compute_points(code_path: Path, p_grid: np.ndarray) -> dict:
    code = load_code(code_path)
    basis = build_x_observation_basis(code.hx, code.hz)
    points: list[dict[str, float]] = []

    for p in p_grid:
        if p == 0.0:
            h_s = h_q = h_s_prime = h_q_prime = 0.0
        else:
            dist_q, deriv_q = xor_distribution_and_derivative(
                basis.column_masks,
                basis.quotient_rank,
                float(p),
            )
            dist_s, deriv_s = marginalize_to_syndrome(
                dist_q,
                deriv_q,
                basis.syndrome_rank,
                basis.quotient_rank,
            )
            h_q, h_q_prime = entropy_and_derivative(dist_q, deriv_q)
            h_s, h_s_prime = entropy_and_derivative(dist_s, deriv_s)

        raw_error = code.n * binary_entropy(float(p)) - h_s
        class_entropy = h_q - h_s
        saved = raw_error - class_entropy
        class_derivative = h_q_prime - h_s_prime
        if p == 0.0:
            class_derivative = 0.0

        points.append(
            {
                "p": float(p),
                "exact_x_error": raw_error,
                "exact_x_class": class_entropy,
                "exact_x_saved_by_stabilizers": saved,
                "exact_x_class_component_norm": class_entropy / code.n,
                "exact_x_class_component_norm_dp": class_derivative / code.n,
            }
        )

    derivative_values = np.array(
        [max(0.0, point["exact_x_class_component_norm_dp"]) for point in points],
        dtype=float,
    )
    peak_idx = int(np.argmax(derivative_values))
    peak_value = float(derivative_values[peak_idx])
    scale = 1.0 / peak_value if peak_value > 0.0 else 1.0
    for point in points:
        point["scaled_exact_x_class_component_norm_dp"] = (
            point["exact_x_class_component_norm_dp"] * scale
        )

    area = float(np.trapezoid(derivative_values, p_grid))
    return {
        "code": {
            "path": str(code_path),
            "name": code.name,
            "n": code.n,
            "k": code.k,
            "rank_hx": code.rank_hx,
            "rank_hz": code.rank_hz,
        },
        "channel": "BSC bit-flip component",
        "component": "x",
        "observation_basis": {
            "syndrome_rank": basis.syndrome_rank,
            "quotient_rank": basis.quotient_rank,
        },
        "scaling": {
            "peak_p": float(points[peak_idx]["p"]),
            "peak_gexit": peak_value,
            "scale_to_unit_peak": scale,
            "trapezoid_area_component_norm": area,
            "target_area_k_over_n": code.k / code.n,
        },
        "points": points,
    }


def tikz_coordinates(points: list[dict], key: str) -> str:
    return " ".join(f"({point['p']:.12g},{point[key]:.12g})" for point in points)


def write_csv(result: dict, out_path: Path) -> None:
    fields = [
        "p",
        "exact_x_error",
        "exact_x_class",
        "exact_x_saved_by_stabilizers",
        "exact_x_class_component_norm",
        "exact_x_class_component_norm_dp",
        "scaled_exact_x_class_component_norm_dp",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for point in result["points"]:
            writer.writerow({field: point[field] for field in fields})


def write_json(result: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def plot_bit_scale(result: dict, out_path: Path) -> None:
    points = result["points"]
    p = np.array([point["p"] for point in points], dtype=float)
    code = result["code"]
    series = [
        ("exact_x_error", r"$\mathbb{E}H(x\mid S)$"),
        ("exact_x_class", r"$\mathbb{E}H(C_X\mid S)$"),
        ("exact_x_saved_by_stabilizers", r"$\mathbb{E}H(x\mid C_X,S)$"),
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
    ax.set_title(rf"{code['name'].replace('_HxHzLxLz', '')}: BSC bit scale")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_bit_scale_entropy_axis(result: dict, out_path: Path) -> None:
    points = result["points"]
    p = np.array([point["p"] for point in points], dtype=float)
    t = binary_entropy_axis(p)
    code = result["code"]
    series = [
        ("exact_x_error", r"$\mathbb{E}H(x\mid S)$"),
        ("exact_x_class", r"$\mathbb{E}H(C_X\mid S)$"),
        ("exact_x_saved_by_stabilizers", r"$\mathbb{E}H(x\mid C_X,S)$"),
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
    ax.set_title(rf"{code['name'].replace('_HxHzLxLz', '')}: BSC bit scale")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_scaled_gexit(result: dict, out_path: Path) -> None:
    points = result["points"]
    p = np.array([point["p"] for point in points], dtype=float)
    y = np.array(
        [point["scaled_exact_x_class_component_norm_dp"] for point in points],
        dtype=float,
    )
    scaling = result["scaling"]
    fig, ax = plt.subplots(figsize=(6.1, 4.1))
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
    ax.axvline(0.5, color="black", linestyle=":", linewidth=0.9)
    ax.plot(
        p,
        y,
        marker=".",
        linewidth=1.45,
        label=(
            rf"$d=5$, scaled by "
            rf"${format_scale(scaling['scale_to_unit_peak'])}$"
        ),
    )
    ax.set_xlabel(r"BSC crossover probability $p$")
    ax.set_ylabel(r"$\widetilde g_X^{\rm BSC}(p)$")
    ax.set_title("Surface-code BSC GEXIT derivative")
    ax.set_xlim(0.0, 0.5)
    ax.set_ylim(-0.02, 1.06)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_scaled_gexit_entropy_axis(result: dict, out_path: Path) -> None:
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
        label=rf"$d=5$, scaled by ${format_scale(scale)}$",
    )
    ax.set_xlabel(r"BSC channel entropy $t=h_2(p)$")
    ax.set_ylabel(r"$\widetilde g_X^{\rm BSC}(t)$")
    ax.set_title("Surface-code BSC GEXIT derivative")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.02, 1.06)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def format_scale(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) >= 1000.0 or abs(value) < 0.01:
        exponent = int(math.floor(math.log10(abs(value))))
        mantissa = value / (10.0**exponent)
        return rf"{mantissa:.3g}\times 10^{{{exponent}}}"
    return f"{value:.3g}"


def write_bit_scale_tikz(result: dict, out_path: Path) -> None:
    points = result["points"]
    code = result["code"]
    max_y = max(float(point["exact_x_error"]) for point in points)
    ymax = max(1.05, max_y * 1.05)
    series = [
        ("exact_x_error", r"$\mathbb{E}H(x\mid S)$", "blue", "*"),
        ("exact_x_class", r"$\mathbb{E}H(C_X\mid S)$", "orange", "square*"),
        (
            "exact_x_saved_by_stabilizers",
            r"$\mathbb{E}H(x\mid C_X,S)$",
            "green!60!black",
            "triangle*",
        ),
    ]
    lines = [
        "% Compact PGFPlots panel for the surface5 BSC bit-scale decomposition.",
        "% Requires \\usepackage{pgfplots} and \\pgfplotsset{compat=1.18}.",
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        "  name=surfaceFiveBSCBitScaleAxis,",
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
        rf"\addplot[black!70, dashed, line width=0.55pt] coordinates {{(0,{code['k']:.12g}) (0.5,{code['k']:.12g})}};",
        r"\addlegendentry{$k$}",
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_scaled_gexit_tikz(result: dict, out_path: Path) -> None:
    scale = result["scaling"]["scale_to_unit_peak"]
    lines = [
        "% Compact PGFPlots panel for the surface5 BSC GEXIT derivative.",
        "% Requires \\usepackage{pgfplots} and \\pgfplotsset{compat=1.18}.",
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        "  name=scaledBSCGEXITSurfaceFiveAxis,",
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
        rf"coordinates {{{tikz_coordinates(result['points'], 'scaled_exact_x_class_component_norm_dp')}}};",
        rf"\addlegendentry{{$d=5$, scaled by ${format_scale(scale)}$}}",
        "\\end{axis}",
        "\\end{tikzpicture}",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    code_path = args.code.resolve()
    out_dir = args.out_dir.resolve()
    tikz_dir = args.tikz_dir.resolve()
    p_grid = np.linspace(0.0, 0.5, int(args.points))

    result = compute_points(code_path, p_grid)
    stem = f"{code_path.stem}_bsc_gexit"
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"
    bit_png = out_dir / f"{code_path.stem}_bsc_bit_scale.png"
    bit_entropy_png = out_dir / f"{code_path.stem}_bsc_bit_scale_entropy_axis.png"
    gexit_png = out_dir / f"{code_path.stem}_scaled_bsc_gexit.png"
    gexit_entropy_png = out_dir / f"{code_path.stem}_scaled_bsc_gexit_entropy_axis.png"
    bit_tikz = tikz_dir / "fig_surface5_bsc_bit_scale_compact.tex"
    gexit_tikz = tikz_dir / "fig_scaled_bsc_gexit_surface5_compact.tex"

    write_json(result, json_path)
    write_csv(result, csv_path)
    plot_bit_scale(result, bit_png)
    plot_bit_scale_entropy_axis(result, bit_entropy_png)
    plot_scaled_gexit(result, gexit_png)
    plot_scaled_gexit_entropy_axis(result, gexit_entropy_png)
    write_bit_scale_tikz(result, bit_tikz)
    write_scaled_gexit_tikz(result, gexit_tikz)

    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {bit_png}")
    print(f"wrote {bit_entropy_png}")
    print(f"wrote {gexit_png}")
    print(f"wrote {gexit_entropy_png}")
    print(f"wrote {bit_tikz}")
    print(f"wrote {gexit_tikz}")
    print(
        "area check: "
        f"{result['scaling']['trapezoid_area_component_norm']:.8g} vs "
        f"k/n={result['scaling']['target_area_k_over_n']:.8g}; "
        f"peak at p={result['scaling']['peak_p']:.4g}"
    )


if __name__ == "__main__":
    main()
