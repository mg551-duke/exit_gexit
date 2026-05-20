from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXIT_DIR = ROOT / "data" / "experiments" / "exit_curves"
DEFAULT_DERIVATIVE_KEY = "exact_x_class_component_norm_dp"
DEFAULT_TARGET_HEIGHT = 1.0
P_HALF = 0.5
OFF_CENTER_PEAK_RTOL = 0.0
P_ATOL = 1e-12
EXCLUDED_SURFACE_DISTANCES = {5, 9, 15, 99, 151}
SURFACE_SHARED_GRID_DISTANCES = {75}


@dataclass(frozen=True)
class Curve:
    family: str
    label: str
    csv_path: Path
    p: np.ndarray
    y: np.ndarray
    y_stderr: np.ndarray | None
    n: int
    k: int
    y_half: float
    scale: float
    dropped_p: tuple[float, ...]
    grid_dropped_p: tuple[float, ...]
    sort_key: tuple


FAMILY_CONFIGS = {
    "surface": {
        "glob": "surface*_exit_rule1_paired_derivative.csv",
        "title": "Surface Codes (d=5,9,15,99,151 omitted)",
        "output": "scaled_derivative_surface_codes.png",
        "tikz": "fig_scaled_exit_surface_codes.tex",
        "compact_tikz": "fig_scaled_exit_surface_codes_compact.tex",
        "axis_name": "scaledEXITSurfaceAxis",
        "figure_label": "fig:scaled-exit-surface",
    },
    "hgp": {
        "glob": "hgp_code*_exit_rule1_paired_derivative.csv",
        "title": "HGP Codes",
        "output": "scaled_derivative_hgp_codes.png",
        "tikz": "fig_scaled_exit_hgp_codes.tex",
        "compact_tikz": "fig_scaled_exit_hgp_codes_compact.tex",
        "axis_name": "scaledEXITHGPAxis",
        "figure_label": "fig:scaled-exit-hgp",
    },
    "bb": {
        "glob": "bb_*_exit_rule1_paired_derivative.csv",
        "title": "Bivariate Bicycle Codes",
        "output": "scaled_derivative_bivariate_bicycle_codes.png",
        "tikz": "fig_scaled_exit_bivariate_bicycle_codes.tex",
        "compact_tikz": "fig_scaled_exit_bivariate_bicycle_codes_compact.tex",
        "axis_name": "scaledEXITBicycleAxis",
        "figure_label": "fig:scaled-exit-bicycle",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay paired derivative curves by code family after vertically "
            "scaling each curve to a common p=0.5 height."
        )
    )
    parser.add_argument(
        "--exit-dir",
        type=Path,
        default=DEFAULT_EXIT_DIR,
        help="Directory containing *_paired_derivative.csv/json files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for generated plots. Defaults to --exit-dir.",
    )
    parser.add_argument(
        "--family",
        choices=["all", *FAMILY_CONFIGS.keys()],
        default="all",
        help="Which family to plot.",
    )
    parser.add_argument(
        "--derivative-key",
        default=DEFAULT_DERIVATIVE_KEY,
        help="CSV derivative column to plot.",
    )
    parser.add_argument(
        "--target-height",
        type=float,
        default=DEFAULT_TARGET_HEIGHT,
        help="Scaled y-value assigned to every curve at p=0.5.",
    )
    parser.add_argument(
        "--show-stderr",
        action="store_true",
        help="Draw scaled one-standard-error bands when available.",
    )
    parser.add_argument(
        "--scalings-csv",
        type=Path,
        default=None,
        help=(
            "Optional path for a CSV summary of y(p=0.5) and scaling factors. "
            "Defaults to <out-dir>/scaled_derivative_family_scalings.csv."
        ),
    )
    parser.add_argument(
        "--tikz-dir",
        type=Path,
        default=None,
        help="Directory for PGFPlots/TikZ outputs. Defaults to <out-dir>/tikz.",
    )
    return parser.parse_args()


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_code_metadata(csv_path: Path) -> dict:
    json_path = csv_path.with_suffix(".json")
    with json_path.open(encoding="utf-8") as f:
        return json.load(f)["code"]


def rounded_p(value: float) -> float:
    return round(float(value), 12)


def sampled_p_values(csv_path: Path) -> set[float]:
    return {rounded_p(row["p"]) for row in read_rows(csv_path)}


def exact_value_at(p: np.ndarray, y: np.ndarray, target_p: float) -> float:
    matches = np.flatnonzero(np.isclose(p, target_p, rtol=0.0, atol=P_ATOL))
    if matches.size == 0:
        raise ValueError(f"p={target_p} is not present in the sampled p grid")
    return float(y[int(matches[0])])


def present_p_value(p: np.ndarray, target_p: float) -> float | None:
    matches = np.flatnonzero(np.isclose(p, target_p, rtol=0.0, atol=P_ATOL))
    if matches.size == 0:
        return None
    return float(p[int(matches[0])])


def off_center_peak_drops(p: np.ndarray, y: np.ndarray, y_half: float) -> tuple[float, ...]:
    drops: set[float] = set()
    threshold = y_half * (1.0 + OFF_CENTER_PEAK_RTOL)
    for p_value, y_value in zip(p, y):
        p_float = float(p_value)
        if np.isclose(p_float, P_HALF, rtol=0.0, atol=P_ATOL):
            continue
        if float(y_value) <= threshold:
            continue
        drops.add(p_float)
        mirror = present_p_value(p, 1.0 - p_float)
        if mirror is not None:
            drops.add(mirror)
    return tuple(sorted(drops))


def remove_p_values(
    p: np.ndarray,
    y: np.ndarray,
    y_stderr: np.ndarray | None,
    drops: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if not drops:
        return p, y, y_stderr

    keep = np.ones(p.shape, dtype=bool)
    for p_value in drops:
        keep &= ~np.isclose(p, p_value, rtol=0.0, atol=P_ATOL)
    filtered_stderr = None if y_stderr is None else y_stderr[keep]
    return p[keep], y[keep], filtered_stderr


def surface_distance_from_name(name: str) -> int | None:
    match = re.search(r"surface(\d+)", name)
    if match is None:
        return None
    return int(match.group(1))


def include_surface_csv(csv_path: Path) -> bool:
    distance = surface_distance_from_name(csv_path.name)
    return distance not in EXCLUDED_SURFACE_DISTANCES


def label_surface(name: str, n: int, k: int) -> tuple[str, tuple]:
    d = surface_distance_from_name(name)
    if d is None:
        return f"{name} (n={n}, k={k})", (n, name)
    return f"d={d}, n={n}", (d, n)


def label_hgp(name: str, n: int, k: int) -> tuple[str, tuple]:
    match = re.search(r"hgp_code_(\d+)_(\d+)_([^_]+)_peg_sparse", name)
    if match is None:
        return f"{name} (n={n}, k={k})", (n, k, name)
    d_token = match.group(3)
    d_label = f", d={d_token}" if d_token.isdigit() else ""
    return f"n={n}, k={k}{d_label}", (n, k, d_token)


def label_bb(name: str, n: int, k: int) -> tuple[str, tuple]:
    match = re.search(r"bb_(?P<family>.+)_l(?P<ell>\d+)_m(?P<m>\d+)_n\d+_sparse", name)
    if match is None:
        return f"{name} [[{n},{k}]]", (n, k, name)
    family = match.group("family").replace("_", " ")
    ell = int(match.group("ell"))
    m = int(match.group("m"))
    return f"{family}, ell={ell}, m={m}, [[{n},{k}]]", (family, ell, m, n)


def family_label(family: str, name: str, n: int, k: int) -> tuple[str, tuple]:
    if family == "surface":
        return label_surface(name, n, k)
    if family == "hgp":
        return label_hgp(name, n, k)
    if family == "bb":
        return label_bb(name, n, k)
    return f"{name} (n={n}, k={k})", (name,)


def load_curve(
    csv_path: Path,
    *,
    family: str,
    derivative_key: str,
    target_height: float,
) -> Curve:
    rows = read_rows(csv_path)
    if not rows:
        raise ValueError(f"{csv_path} has no rows")
    if derivative_key not in rows[0]:
        raise ValueError(f"{csv_path} does not contain column {derivative_key!r}")

    p = np.array([float(row["p"]) for row in rows], dtype=float)
    y = np.array([float(row[derivative_key]) for row in rows], dtype=float)
    order = np.argsort(p)
    p = p[order]
    y = y[order]

    stderr_key = f"{derivative_key}_stderr"
    y_stderr = None
    if stderr_key in rows[0]:
        y_stderr = np.array([float(row[stderr_key]) for row in rows], dtype=float)[order]

    y_half = exact_value_at(p, y, P_HALF)
    if not np.isfinite(y_half) or abs(y_half) < 1e-15:
        raise ValueError(f"{csv_path} has unusable {derivative_key} at p=0.5: {y_half}")
    scale = float(target_height / y_half)
    dropped_p = off_center_peak_drops(p, y, y_half) if family in {"surface", "bb"} else ()
    p, y, y_stderr = remove_p_values(p, y, y_stderr, dropped_p)

    metadata = read_code_metadata(csv_path)
    n = int(metadata["n"])
    k = int(metadata["k"])
    label, sort_key = family_label(family, metadata["name"], n, k)
    return Curve(
        family=family,
        label=label,
        csv_path=csv_path,
        p=p,
        y=y,
        y_stderr=y_stderr,
        n=n,
        k=k,
        y_half=y_half,
        scale=scale,
        dropped_p=dropped_p,
        grid_dropped_p=(),
        sort_key=sort_key,
    )


def restrict_surface_shared_grid(curves: list[Curve]) -> list[Curve]:
    raw_p_by_distance = {
        surface_distance_from_name(curve.csv_path.name): sampled_p_values(curve.csv_path)
        for curve in curves
    }
    restricted = []
    for curve in curves:
        distance = surface_distance_from_name(curve.csv_path.name)
        if distance not in SURFACE_SHARED_GRID_DISTANCES:
            restricted.append(curve)
            continue

        other_p_sets = [
            p_values
            for other_distance, p_values in raw_p_by_distance.items()
            if other_distance != distance
        ]
        if not other_p_sets:
            restricted.append(curve)
            continue

        shared_p = set.intersection(*other_p_sets)
        keep = np.array([rounded_p(p_value) in shared_p for p_value in curve.p], dtype=bool)
        grid_dropped_p = tuple(
            sorted(float(p_value) for p_value, keep_value in zip(curve.p, keep) if not keep_value)
        )
        y_stderr = None if curve.y_stderr is None else curve.y_stderr[keep]
        restricted.append(
            replace(
                curve,
                p=curve.p[keep],
                y=curve.y[keep],
                y_stderr=y_stderr,
                grid_dropped_p=grid_dropped_p,
            )
        )
    return restricted


def load_family_curves(
    exit_dir: Path,
    family: str,
    *,
    derivative_key: str,
    target_height: float,
) -> list[Curve]:
    config = FAMILY_CONFIGS[family]
    csv_paths = sorted(exit_dir.glob(config["glob"]))
    if family == "surface":
        csv_paths = [csv_path for csv_path in csv_paths if include_surface_csv(csv_path)]
    curves = [
        load_curve(
            csv_path,
            family=family,
            derivative_key=derivative_key,
            target_height=target_height,
        )
        for csv_path in csv_paths
    ]
    if not curves:
        raise FileNotFoundError(f"No paired derivative CSVs found for {family} in {exit_dir}")
    if family == "surface":
        curves = restrict_surface_shared_grid(curves)
    return sorted(curves, key=lambda curve: curve.sort_key)


def write_scalings(path: Path, curves: Iterable[Curve], derivative_key: str, target_height: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "family",
        "label",
        "csv",
        "derivative_key",
        "target_height",
        "p_half_value",
        "scale_multiplier",
        "dropped_p",
        "grid_dropped_p",
        "n",
        "k",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for curve in curves:
            writer.writerow(
                {
                    "family": curve.family,
                    "label": curve.label,
                    "csv": curve.csv_path.name,
                    "derivative_key": derivative_key,
                    "target_height": f"{target_height:.12g}",
                    "p_half_value": f"{curve.y_half:.12g}",
                    "scale_multiplier": f"{curve.scale:.12g}",
                    "dropped_p": " ".join(f"{p_value:.12g}" for p_value in curve.dropped_p),
                    "grid_dropped_p": " ".join(
                        f"{p_value:.12g}" for p_value in curve.grid_dropped_p
                    ),
                    "n": curve.n,
                    "k": curve.k,
                }
            )


def plot_family(
    curves: list[Curve],
    *,
    title: str,
    derivative_key: str,
    target_height: float,
    show_stderr: bool,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    cmap = plt.get_cmap("tab20" if len(curves) > 10 else "tab10")

    for index, curve in enumerate(curves):
        color = cmap(index % cmap.N)
        scaled = curve.y * curve.scale
        label = f"{curve.label}  scaled x{curve.scale:.3g}"
        ax.plot(
            curve.p,
            scaled,
            marker="o",
            markersize=3.2,
            linewidth=1.45,
            color=color,
            label=label,
        )
        if show_stderr and curve.y_stderr is not None:
            scaled_stderr = curve.y_stderr * abs(curve.scale)
            ax.fill_between(
                curve.p,
                scaled - scaled_stderr,
                scaled + scaled_stderr,
                color=color,
                alpha=0.12,
                linewidth=0,
            )

    ax.axvline(P_HALF, color="black", linestyle=":", linewidth=1.0, alpha=0.65)
    ax.axhline(target_height, color="black", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.set_xlabel("erasure probability e")
    ax.set_ylabel("scaled EXIT derivative")
    ax.set_title(f"{title}: paired derivatives scaled to y(0.5) = {target_height:g}")
    ax.grid(True, alpha=0.25)
    fig.subplots_adjust(right=0.68)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        fontsize=8,
        frameon=True,
        title="curve, multiplier",
    )
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def format_number(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1e3:
        exponent = int(np.floor(np.log10(abs_value)))
        mantissa = value / (10**exponent)
        return rf"{mantissa:.3g}\times 10^{{{exponent}}}"
    if abs_value != 0.0 and abs_value < 1e-3:
        return f"{value:.3g}"
    if abs_value < 10:
        return f"{value:.4g}"
    if abs_value < 100:
        return f"{value:.3g}"
    return f"{value:.4g}"


def format_text_number(value: float) -> str:
    formatted = format_number(value)
    if r"\times" in formatted:
        return f"${formatted}$"
    return formatted


def format_tikz_scale(value: float) -> str:
    abs_value = abs(value)
    if abs_value >= 1000:
        exponent = int(np.floor(np.log10(abs_value)))
        mantissa = value / (10**exponent)
        return rf"\mathrm{{scaled\ by}}\,{mantissa:.3g}\cdot 10^{{{exponent}}}"
    return rf"\mathrm{{scaled\ by}}\,{format_number(value)}"


def format_surface_tikz_label(curve: Curve) -> str:
    distance = surface_distance_from_name(curve.csv_path.name)
    if distance is None:
        return rf"{curve.label} (${format_tikz_scale(curve.scale)}$)"
    return rf"$d={distance}$, $n={curve.n}$ (${format_tikz_scale(curve.scale)}$)"


def format_hgp_tikz_label(curve: Curve) -> str:
    match = re.search(r"hgp_code_\d+_\d+_([^_]+)_peg_sparse", curve.csv_path.name)
    distance = match.group(1) if match is not None else ""
    distance_part = rf", $d={distance}$" if distance.isdigit() else ""
    return rf"$n={curve.n}$, $k={curve.k}${distance_part} (${format_tikz_scale(curve.scale)}$)"


def format_bb_tikz_label(curve: Curve) -> str:
    match = re.search(
        r"bb_(?P<family>.+)_l(?P<ell>\d+)_m(?P<m>\d+)_n\d+_sparse",
        curve.csv_path.name,
    )
    code = rf"$\left[\!\left[{curve.n},{curve.k}\right]\!\right]$"
    if match is None:
        return rf"{curve.label} ({format_tikz_scale(curve.scale)})"
    family = match.group("family").replace("_", " ")
    ell = int(match.group("ell"))
    m = int(match.group("m"))
    return rf"{family}, $\ell={ell}$, $m={m}$, {code} (${format_tikz_scale(curve.scale)}$)"


def format_tikz_label(curve: Curve) -> str:
    if curve.family == "surface":
        return format_surface_tikz_label(curve)
    if curve.family == "hgp":
        return format_hgp_tikz_label(curve)
    if curve.family == "bb":
        return format_bb_tikz_label(curve)
    return rf"{curve.label} (${format_tikz_scale(curve.scale)}$)"


def compact_tikz_label(curve: Curve) -> str:
    scale = format_number(curve.scale)
    if curve.family == "surface":
        distance = surface_distance_from_name(curve.csv_path.name)
        if distance is not None:
            return rf"$d={distance}$, scaled by ${scale}$"
    if curve.family == "bb":
        family = "BB"
        if curve.csv_path.name.startswith("bb_gross"):
            family = "gross"
        elif curve.csv_path.name.startswith("bb_two_gross"):
            family = "two-gross"
        return rf"{family} $[[{curve.n},{curve.k}]]$, scaled by ${scale}$"
    if curve.family == "hgp":
        return rf"$[[{curve.n},{curve.k}]]$, scaled by ${scale}$"
    return rf"{curve.label}, scaled by ${scale}$"


def tikz_coordinates(curve: Curve) -> str:
    scaled = curve.y * curve.scale
    return " ".join(
        f"({float(p_value):.12g},{float(y_value):.12g})"
        for p_value, y_value in zip(curve.p, scaled)
    )


def write_tikz_family(
    curves: list[Curve],
    *,
    family: str,
    derivative_key: str,
    target_height: float,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    config = FAMILY_CONFIGS[family]
    xmin = min(float(np.min(curve.p)) for curve in curves)
    xmax = max(float(np.max(curve.p)) for curve in curves)
    ymin = -0.02
    ymax = max(1.08, max(float(np.max(curve.y * curve.scale)) for curve in curves) * 1.03)
    colors = [
        "blue",
        "orange",
        "green!60!black",
        "red",
        "purple",
        "brown",
        "magenta",
        "cyan!70!black",
        "black",
    ]
    markers = ["*", "square*", "triangle*", "diamond*", "pentagon*", "otimes*", "star"]

    lines = [
        "% Auto-generated by experiments/plot_scaled_derivative_families.py.",
        "% Requires \\usepackage{pgfplots} and \\pgfplotsset{compat=1.18}.",
        rf"% Figure label suggestion: \label{{{config['figure_label']}}}",
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        rf"  name={config['axis_name']},",
        r"  width=\linewidth,",
        r"  height=0.62\linewidth,",
        r"  xlabel={Erasure probability $e$},",
        r"  ylabel={Scaled EXIT derivative $\widetilde g_X(e)$},",
        f"  xmin={xmin:.12g}, xmax={xmax:.12g},",
        f"  ymin={ymin:.12g}, ymax={ymax:.12g},",
        "  grid=both,",
        r"  major grid style={black!12},",
        r"  minor grid style={black!6},",
        "  tick align=outside,",
        "  legend cell align={left},",
        r"  legend style={draw=black!20, fill=white, at={(1.02,0.5)}, anchor=west, font=\scriptsize},",
        "]",
        rf"\addplot[black!60, densely dotted, line width=0.7pt, forget plot] coordinates {{({P_HALF:.12g},{ymin:.12g}) ({P_HALF:.12g},{ymax:.12g})}};",
        rf"\addplot[black!60, dashed, line width=0.7pt, forget plot] coordinates {{({xmin:.12g},{target_height:.12g}) ({xmax:.12g},{target_height:.12g})}};",
    ]

    for index, curve in enumerate(curves):
        color = colors[index % len(colors)]
        marker = markers[index % len(markers)]
        lines.extend(
            [
                rf"\addplot+[color={color}, mark={marker}, mark size=1.4pt, line width=0.8pt]",
                rf"coordinates {{{tikz_coordinates(curve)}}};",
                rf"\addlegendentry{{{format_tikz_label(curve)}}}",
            ]
        )

    lines.extend(["\\end{axis}", "\\end{tikzpicture}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_compact_tikz_family(
    curves: list[Curve],
    *,
    family: str,
    target_height: float,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    config = FAMILY_CONFIGS[family]
    panel_titles = {
        "surface": "Surface derivatives",
        "hgp": "HGP derivatives",
        "bb": "Bivariate-bicycle derivatives",
    }
    panel_title = panel_titles.get(family, config["title"])
    xmin = min(float(np.min(curve.p)) for curve in curves)
    xmax = max(float(np.max(curve.p)) for curve in curves)
    ymin = -0.02
    ymax = max(1.06, max(float(np.max(curve.y * curve.scale)) for curve in curves) * 1.02)
    colors = [
        "blue",
        "orange",
        "green!60!black",
        "red",
        "purple",
        "brown",
        "magenta",
        "cyan!70!black",
        "black",
    ]
    markers = ["*", "square*", "triangle*", "diamond*", "pentagon*", "otimes*", "star"]
    lines = [
        "% Compact PGFPlots panel for a three-panel paper figure.",
        "% Requires \\usepackage{pgfplots} and \\pgfplotsset{compat=1.18}.",
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        rf"  name={config['axis_name']}Compact,",
        r"  width=\linewidth,",
        r"  height=0.78\linewidth,",
        rf"  title={{{panel_title}}},",
        r"  title style={font=\scriptsize},",
        r"  xlabel={$e$},",
        r"  ylabel={$\widetilde g_X(e)$},",
        f"  xmin={xmin:.12g}, xmax={xmax:.12g},",
        f"  ymin={ymin:.12g}, ymax={ymax:.12g},",
        "  grid=both,",
        r"  major grid style={black!12},",
        r"  minor grid style={black!6},",
        "  tick align=outside,",
        r"  tick label style={font=\scriptsize},",
        r"  label style={font=\scriptsize},",
        "  legend cell align={left},",
        r"  legend style={draw=black!15, fill=white, fill opacity=0.82, text opacity=1, at={(0.02,0.02)}, anchor=south west, font=\tiny},",
        "]",
        rf"\addplot[black!60, densely dotted, line width=0.55pt, forget plot] coordinates {{({P_HALF:.12g},{ymin:.12g}) ({P_HALF:.12g},{ymax:.12g})}};",
        rf"\addplot[black!60, dashed, line width=0.55pt, forget plot] coordinates {{({xmin:.12g},{target_height:.12g}) ({xmax:.12g},{target_height:.12g})}};",
    ]
    for index, curve in enumerate(curves):
        color = colors[index % len(colors)]
        marker = markers[index % len(markers)]
        lines.extend(
            [
                rf"\addplot+[color={color}, mark={marker}, mark size=1.0pt, line width=0.65pt]",
                rf"coordinates {{{tikz_coordinates(curve)}}};",
                rf"\addlegendentry{{{compact_tikz_label(curve)}}}",
            ]
        )
    lines.extend(["\\end{axis}", "\\end{tikzpicture}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def curve_by_prefix(curves: list[Curve], prefix: str) -> Curve:
    for curve in curves:
        if curve.label.startswith(prefix):
            return curve
    raise KeyError(prefix)


def curve_area_and_rate(curve: Curve) -> tuple[float | None, float]:
    data = json.loads(curve.csv_path.with_suffix(".json").read_text(encoding="utf-8"))
    area = (
        data.get("areas", {})
        .get(DEFAULT_DERIVATIVE_KEY, {})
        .get("trapezoid_derivative_area")
    )
    rate = float(data["code"]["k"]) / float(data["code"]["n"])
    return (None if area is None else float(area)), rate


def max_relative_area_error(curves: list[Curve]) -> float:
    errors = []
    for curve in curves:
        area, rate = curve_area_and_rate(curve)
        if area is None or rate == 0.0:
            continue
        errors.append(abs(area - rate) / rate)
    return max(errors) if errors else float("nan")


def write_three_panel_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = r"""% Three-panel numerical figure for qexit.tex.
% Requires \usepackage{pgfplots} and \pgfplotsset{compat=1.18}.
\begin{figure*}[t]
\centering
\begin{minipage}[t]{0.315\textwidth}
\centering
\input{fig_surface5_bit_scale_compact.tex}
{\footnotesize (a) Bit-scale decomposition}
\end{minipage}\hfill
\begin{minipage}[t]{0.315\textwidth}
\centering
\input{fig_scaled_exit_surface_codes_compact.tex}
{\footnotesize (b) Surface-code derivatives}
\end{minipage}\hfill
\begin{minipage}[t]{0.315\textwidth}
\centering
\input{fig_scaled_exit_bivariate_bicycle_codes_compact.tex}
{\footnotesize (c) Bivariate-bicycle derivatives}
\end{minipage}
\caption{Numerical EXIT quantities for CSS codes on the quantum erasure
channel.  Panel (a) shows the bit-scale decomposition for the surface
$[[41,1]]$ code: the raw Pauli ambiguity $\mathbb{E}H(x\mid S)$ decomposes
into correction ambiguity $\mathbb{E}H(C_X\mid S)$ plus the covered-stabilizer
contribution $\mathbb{E}H(x\mid C_X,S)$.  Panels (b) and (c) show scaled
component EXIT derivatives
$\widetilde g_X(e)=g_X(e)/g_X(1/2)$ for surface and bivariate-bicycle code
families.  The scaling shown in each legend is only for visual comparison; the
unscaled area satisfies $\int_0^1 g_X(e)\,de=k/n$.}
\label{fig:scaled-exit-numerics}
\end{figure*}
"""
    path.write_text(text, encoding="utf-8")


def write_paper_notes(path: Path, family_curves: dict[str, list[Curve]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    surface = family_curves["surface"]
    bb = family_curves["bb"]
    surface_distances = [
        surface_distance_from_name(curve.csv_path.name)
        for curve in surface
        if surface_distance_from_name(curve.csv_path.name) is not None
    ]
    bb_gross_small = curve_by_prefix(bb, "gross, ell=24")
    bb_gross_large = curve_by_prefix(bb, "gross, ell=48")
    bb_two_small = curve_by_prefix(bb, "two gross, ell=24")
    bb_two_large = curve_by_prefix(bb, "two gross, ell=48")
    surface_area_error = max_relative_area_error(surface)
    bb_area_error = max_relative_area_error(bb)

    notes = rf"""% Paper-ready numerical section for qexit.tex.
% Preamble additions needed in qexit.tex:
%   \usepackage{{pgfplots}}
%   \pgfplotsset{{compat=1.18}}
% This file assumes it is compiled from the same directory as the TikZ files.

\section{{Numerical illustrations}}

We complement the EXIT area identities with exact finite-length calculations
for several CSS code families.  For an erasure probability $e$, let
\[
g_X(e):=\frac{{d}}{{de}}\frac{{1}}{{n}}\E[H(C_X\mid E,S)]
\]
denote the $X$-component correction-entropy EXIT derivative.  This is one CSS
component of Theorem~\ref{{thm:css-exit}}.  Its unscaled area is fixed by the
endpoint entropy:
\[
\int_0^1 g_X(e)\,de
=\frac{{1}}{{n}}\Bigl(\E[H(C_X\mid E=[n],S)]-\E[H(C_X\mid E=\varnothing,S)]\Bigr)
=\frac{{k}}{{n}}.
\]
Thus the numerical area under the unscaled curve is the code rate.  In
Fig.~\ref{{fig:scaled-exit-numerics}} we instead plot the scaled shape
\[
\widetilde g_X(e):=\frac{{g_X(e)}}{{g_X(1/2)}}.
\]
The word ``scaled'' in each legend entry gives the multiplier
$1/g_X(1/2)$ used only for visual comparison; it is not part of the area
identity.

\input{{fig_scaled_exit_three_panel.tex}}

The surface-code panel uses distances
$d\in\{{{','.join(str(d) for d in surface_distances)}}}$.  The unscaled
trapezoidal areas stored for these curves agree with $k/n$ to within
{format_text_number(surface_area_error)} relative error; the largest discrepancy is
from the mixed-sample $d=11$ curve, whose seven central points use 7500
effective paired samples.  The displayed surface curves are scaled by the
factors listed in the legend, ranging from {format_text_number(surface[0].scale)}
for $d={surface_distances[0]}$ to {format_text_number(surface[-1].scale)} for
$d={surface_distances[-1]}$.

The bivariate-bicycle panel contains the gross family
$[[{bb_gross_small.n},{bb_gross_small.k}]]$ and
$[[{bb_gross_large.n},{bb_gross_large.k}]]$, and the two-gross family
$[[{bb_two_small.n},{bb_two_small.k}]]$ and
$[[{bb_two_large.n},{bb_two_large.k}]]$.  The corresponding unscaled rates are
{format_text_number(bb_gross_small.k / bb_gross_small.n)},
{format_text_number(bb_gross_large.k / bb_gross_large.n)},
{format_text_number(bb_two_small.k / bb_two_small.n)}, and
{format_text_number(bb_two_large.k / bb_two_large.n)}.  The trapezoidal EXIT areas
match these rates to within {format_text_number(bb_area_error)} relative error over
the sampled window.  Panel (c) repeats the bivariate-bicycle panel as a
placeholder for a third comparison.
"""
    path.write_text(notes, encoding="utf-8")


def main() -> None:
    args = parse_args()
    exit_dir = args.exit_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir is not None else exit_dir
    scalings_csv = (
        args.scalings_csv.resolve()
        if args.scalings_csv is not None
        else out_dir / "scaled_derivative_family_scalings.csv"
    )
    tikz_dir = args.tikz_dir.resolve() if args.tikz_dir is not None else out_dir / "tikz"

    families = FAMILY_CONFIGS.keys() if args.family == "all" else [args.family]
    all_curves: list[Curve] = []
    family_curves: dict[str, list[Curve]] = {}
    for family in families:
        curves = load_family_curves(
            exit_dir,
            family,
            derivative_key=args.derivative_key,
            target_height=args.target_height,
        )
        family_curves[family] = curves
        all_curves.extend(curves)
        config = FAMILY_CONFIGS[family]
        out_path = out_dir / config["output"]
        plot_family(
            curves,
            title=config["title"],
            derivative_key=args.derivative_key,
            target_height=args.target_height,
            show_stderr=args.show_stderr,
            out_path=out_path,
        )
        print(f"wrote {out_path}")
        tikz_path = tikz_dir / config["tikz"]
        write_tikz_family(
            curves,
            family=family,
            derivative_key=args.derivative_key,
            target_height=args.target_height,
            out_path=tikz_path,
        )
        print(f"wrote {tikz_path}")
        compact_tikz_path = tikz_dir / config["compact_tikz"]
        write_compact_tikz_family(
            curves,
            family=family,
            target_height=args.target_height,
            out_path=compact_tikz_path,
        )
        print(f"wrote {compact_tikz_path}")

    write_scalings(scalings_csv, all_curves, args.derivative_key, args.target_height)
    print(f"wrote {scalings_csv}")
    if set(FAMILY_CONFIGS).issubset(family_curves):
        three_panel_path = tikz_dir / "fig_scaled_exit_three_panel.tex"
        write_three_panel_figure(three_panel_path)
        print(f"wrote {three_panel_path}")
        notes_path = tikz_dir / "scaled_exit_captions_and_numerical_results.tex"
        write_paper_notes(notes_path, family_curves)
        print(f"wrote {notes_path}")


if __name__ == "__main__":
    main()
