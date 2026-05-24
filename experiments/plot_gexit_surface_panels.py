from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GEXIT_DIR = ROOT / "data" / "experiments" / "gexit_curves"
DEFAULT_BSC_CLUSTER_DIR = DEFAULT_GEXIT_DIR / "entropy_centered_surface_jobs"
DEFAULT_DEPOLARIZING_DIR = ROOT / "data" / "experiments" / "depolarizing_gexit_curves"
DEFAULT_DEPOLARIZING_CLUSTER_DIR = DEFAULT_DEPOLARIZING_DIR / "cluster_surface_jobs"
DEFAULT_OUT_DIR = DEFAULT_GEXIT_DIR / "surface_gexit_panels"
DEFAULT_TIKZ_DIR = DEFAULT_GEXIT_DIR / "tikz"


@dataclass(frozen=True)
class Curve:
    distance: int
    n: int
    k: int
    csv_path: Path
    t: np.ndarray
    y: np.ndarray
    scale: float
    area: float
    target_area: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build compact surface-code GEXIT panels from pulled BSC and "
            "depolarizing result files."
        )
    )
    parser.add_argument("--bsc-dir", type=Path, default=DEFAULT_BSC_CLUSTER_DIR)
    parser.add_argument(
        "--depolarizing-dir",
        type=Path,
        default=DEFAULT_DEPOLARIZING_CLUSTER_DIR,
        help="Directory containing cluster surface*_depolarizing_gexit_sampled.csv files.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tikz-dir", type=Path, default=DEFAULT_TIKZ_DIR)
    parser.add_argument(
        "--no-include-surface5-bsc",
        dest="include_surface5_bsc",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--no-include-surface5-depolarizing",
        dest="include_surface5_depolarizing",
        action="store_false",
        default=True,
    )
    return parser.parse_args()


def binary_entropy(prob: float) -> float:
    if prob <= 0.0:
        return 0.0
    if prob >= 0.5:
        return 1.0
    return -prob * math.log2(prob) - (1.0 - prob) * math.log2(1.0 - prob)


def binary_entropy_derivative(p: np.ndarray) -> np.ndarray:
    derivative = np.full_like(p, np.nan, dtype=float)
    derivative[p == 0.0] = np.inf
    mask = (p > 0.0) & (p < 0.5)
    derivative[mask] = np.log2((1.0 - p[mask]) / p[mask])
    return derivative


def depolarizing_entropy(prob: float) -> float:
    if prob <= 0.0:
        return 0.0
    if prob >= 0.75:
        return 2.0
    return binary_entropy(prob) + prob * math.log2(3.0)


def depolarizing_entropy_derivative(p: np.ndarray) -> np.ndarray:
    derivative = np.full_like(p, np.nan, dtype=float)
    derivative[p == 0.0] = np.inf
    mask = (p > 0.0) & (p < 0.75)
    derivative[mask] = np.log2(3.0 * (1.0 - p[mask]) / p[mask])
    return derivative


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_metadata(csv_path: Path) -> dict:
    with csv_path.with_suffix(".json").open(encoding="utf-8") as f:
        return json.load(f)


def surface_distance(path: Path, code: dict) -> int:
    if code.get("distance") is not None:
        return int(code["distance"])
    match = re.search(r"surface(\d+)", path.name)
    if match is None:
        raise ValueError(f"could not infer surface distance from {path.name}")
    return int(match.group(1))


def first_existing_key(row: dict[str, str], keys: tuple[str, ...], csv_path: Path) -> str:
    for key in keys:
        if key in row:
            return key
    raise ValueError(f"{csv_path} has none of {keys}")


def finite_curve(
    *,
    distance: int,
    n: int,
    k: int,
    csv_path: Path,
    t: np.ndarray,
    y: np.ndarray,
    target_area: float,
) -> Curve:
    finite = np.isfinite(t) & np.isfinite(y)
    t = t[finite]
    y = y[finite]
    order = np.argsort(t)
    t = t[order]
    y = y[order]
    if t.size == 0:
        raise ValueError(f"{csv_path} has no finite entropy-axis derivative values")
    peak = float(np.max(y))
    scale = 1.0 / peak if peak > 0.0 else 1.0
    area = float(np.trapezoid(y, t))
    return Curve(
        distance=distance,
        n=n,
        k=k,
        csv_path=csv_path,
        t=t,
        y=y,
        scale=scale,
        area=area,
        target_area=target_area,
    )


def load_bsc_curve(csv_path: Path) -> Curve:
    rows = read_rows(csv_path)
    if not rows:
        raise ValueError(f"{csv_path} is empty")
    data = read_metadata(csv_path)
    code = data["code"]
    key = first_existing_key(
        rows[0],
        ("posterior_x_class_component_norm_dp", "exact_x_class_component_norm_dp"),
        csv_path,
    )
    p = np.array([float(row["p"]) for row in rows], dtype=float)
    dydp = np.array([float(row[key]) for row in rows], dtype=float)
    hprime = binary_entropy_derivative(p)
    dydt = np.divide(
        dydp,
        hprime,
        out=np.full_like(dydp, np.nan),
        where=hprime > 0.0,
    )
    dydt[np.isclose(p, 0.5)] = 0.0
    t = np.array([binary_entropy(float(value)) for value in p], dtype=float)
    return finite_curve(
        distance=surface_distance(csv_path, code),
        n=int(code["n"]),
        k=int(code["k"]),
        csv_path=csv_path,
        t=t,
        y=dydt,
        target_area=float(code["k"]) / float(code["n"]),
    )


def load_depolarizing_curve(csv_path: Path) -> Curve:
    rows = read_rows(csv_path)
    if not rows:
        raise ValueError(f"{csv_path} is empty")
    data = read_metadata(csv_path)
    code = data["code"]
    key = first_existing_key(
        rows[0],
        ("posterior_logical_class_component_norm_dp",),
        csv_path,
    )
    p = np.array([float(row["p"]) for row in rows], dtype=float)
    dydp = np.array([float(row[key]) for row in rows], dtype=float)
    t = np.array(
        [
            float(row.get("normalized_channel_entropy") or depolarizing_entropy(float(row["p"])) / 2.0)
            for row in rows
        ],
        dtype=float,
    )
    hprime = depolarizing_entropy_derivative(p)
    dydt = np.divide(
        2.0 * dydp,
        hprime,
        out=np.full_like(dydp, np.nan),
        where=hprime > 0.0,
    )
    dydt[np.isclose(p, 0.75)] = 0.0
    return finite_curve(
        distance=surface_distance(csv_path, code),
        n=int(code["n"]),
        k=int(code["k"]),
        csv_path=csv_path,
        t=t,
        y=dydt,
        target_area=2.0 * float(code["k"]) / float(code["n"]),
    )


def discover_bsc_curves(bsc_dir: Path, include_surface5: bool) -> list[Curve]:
    csv_paths: list[Path] = []
    if include_surface5:
        surface5 = DEFAULT_GEXIT_DIR / "surface5_HxHzLxLz_bsc_gexit.csv"
        if surface5.exists():
            csv_paths.append(surface5)
    csv_paths.extend(sorted(bsc_dir.glob("surface*_bsc_gexit_sampled.csv")))
    curves_by_distance: dict[int, Curve] = {}
    for csv_path in csv_paths:
        curve = load_bsc_curve(csv_path)
        curves_by_distance.setdefault(curve.distance, curve)
    return [curves_by_distance[distance] for distance in sorted(curves_by_distance)]


def discover_depolarizing_curves(
    depolarizing_dir: Path,
    include_surface5: bool,
) -> list[Curve]:
    csv_paths: list[Path] = []
    if include_surface5:
        surface5 = DEFAULT_DEPOLARIZING_DIR / "surface5_depolarizing_gexit_sampled.csv"
        if surface5.exists():
            csv_paths.append(surface5)
    csv_paths.extend(sorted(depolarizing_dir.glob("surface*_depolarizing_gexit_sampled.csv")))
    curves_by_distance: dict[int, Curve] = {}
    for csv_path in csv_paths:
        curve = load_depolarizing_curve(csv_path)
        curves_by_distance[curve.distance] = curve
    return [curves_by_distance[distance] for distance in sorted(curves_by_distance)]


def format_scale(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) >= 1000.0 or abs(value) < 0.01:
        exponent = int(math.floor(math.log10(abs(value))))
        mantissa = value / (10.0**exponent)
        return rf"{mantissa:.3g}\times 10^{{{exponent}}}"
    return f"{value:.3g}"


def tikz_coordinates(curve: Curve) -> str:
    return " ".join(
        f"({x:.12g},{y:.12g})" for x, y in zip(curve.t, curve.y * curve.scale)
    )


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def write_family_tikz(
    curves: list[Curve],
    *,
    out_path: Path,
    axis_name: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    colors = ["blue", "orange", "green!60!black", "red", "purple", "brown"]
    markers = ["*", "square*", "triangle*", "diamond*", "pentagon*", "otimes*"]
    lines = [
        f"% Compact PGFPlots panel for {title}.",
        "% Requires \\usepackage{pgfplots} and \\pgfplotsset{compat=1.18}.",
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        f"  name={axis_name},",
        r"  width=\linewidth,",
        r"  height=0.78\linewidth,",
        rf"  title={{{title}}},",
        r"  title style={font=\scriptsize},",
        rf"  xlabel={{{xlabel}}},",
        rf"  ylabel={{{ylabel}}},",
        "  xmin=0, xmax=1,",
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
        r"\addplot[black!60, dashed, line width=0.55pt, forget plot] coordinates {(0,1) (1,1)};",
    ]
    for idx, curve in enumerate(curves):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        lines.extend(
            [
                rf"\addplot+[color={color}, mark={marker}, mark size=1.0pt, line width=0.65pt]",
                rf"coordinates {{{tikz_coordinates(curve)}}};",
                rf"\addlegendentry{{$d={curve.distance}$, scaled by ${format_scale(curve.scale)}$}}",
            ]
        )
    lines.extend(["\\end{axis}", "\\end{tikzpicture}", ""])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def plot_family_png(curves: list[Curve], *, out_path: Path, title: str, ylabel: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
    ax.axvline(0.5, color="black", linestyle=":", linewidth=0.9)
    for curve in curves:
        ax.plot(
            curve.t,
            curve.y * curve.scale,
            marker=".",
            linewidth=1.35,
            label=rf"$d={curve.distance}$, scaled by ${format_scale(curve.scale)}$",
        )
    ax.set_xlabel("normalized channel entropy")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.02, 1.06)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_scalings(curves: list[Curve], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "distance",
                "n",
                "k",
                "csv",
                "area",
                "target_area",
                "relative_area_error",
                "scale",
            ],
        )
        writer.writeheader()
        for curve in curves:
            error = (
                abs(curve.area - curve.target_area) / curve.target_area
                if curve.target_area
                else float("nan")
            )
            writer.writerow(
                {
                    "distance": curve.distance,
                    "n": curve.n,
                    "k": curve.k,
                    "csv": portable_path(curve.csv_path),
                    "area": f"{curve.area:.12g}",
                    "target_area": f"{curve.target_area:.12g}",
                    "relative_area_error": f"{error:.12g}",
                    "scale": f"{curve.scale:.12g}",
                }
            )


def write_three_panel_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = r"""% Three-panel numerical GEXIT figure.
% Requires \usepackage{pgfplots} and \pgfplotsset{compat=1.18}.
\begin{figure*}[t]
\centering
\begin{minipage}[t]{0.315\textwidth}
\centering
\input{fig_surface5_bsc_bit_scale_compact.tex}
{\footnotesize (a) BSC bit-scale decomposition}
\end{minipage}\hfill
\begin{minipage}[t]{0.315\textwidth}
\centering
\input{fig_scaled_bsc_gexit_surface_codes_compact.tex}
{\footnotesize (b) BSC GEXIT derivatives}
\end{minipage}\hfill
\begin{minipage}[t]{0.315\textwidth}
\centering
\input{fig_scaled_depolarizing_gexit_surface_codes_compact.tex}
{\footnotesize (c) Depolarizing GEXIT derivatives}
\end{minipage}
\caption{Surface-code GEXIT numerics.  Panel (a) shows the BSC bit-scale
decomposition for the $d=5$ surface code.  Panel (b) overlays scaled BSC
component GEXIT derivatives against $t=h_2(p)$.  Panel (c) overlays scaled
depolarizing GEXIT derivatives against normalized Pauli entropy
$t=H_4(p)/2$.  Legend scaling is for visual comparison only; unscaled areas
are recorded in the accompanying scaling CSV files.}
\label{fig:surface-gexit-numerics}
\end{figure*}
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve()
    tikz_dir = args.tikz_dir.resolve()
    bsc_curves = discover_bsc_curves(
        args.bsc_dir.resolve(),
        include_surface5=args.include_surface5_bsc,
    )
    depolarizing_curves = discover_depolarizing_curves(
        args.depolarizing_dir.resolve(),
        include_surface5=args.include_surface5_depolarizing,
    )
    if not bsc_curves:
        raise SystemExit(f"no BSC curves found in {args.bsc_dir}")
    if not depolarizing_curves:
        raise SystemExit(f"no depolarizing curves found in {args.depolarizing_dir}")

    plot_family_png(
        bsc_curves,
        out_path=out_dir / "scaled_bsc_gexit_surface_codes_compact.png",
        title="Surface BSC GEXIT derivatives",
        ylabel=r"$\widetilde g_X^{\rm BSC}(t)$",
    )
    plot_family_png(
        depolarizing_curves,
        out_path=out_dir / "scaled_depolarizing_gexit_surface_codes_compact.png",
        title="Surface depolarizing GEXIT derivatives",
        ylabel=r"$\widetilde g^{\rm depol}(t)$",
    )
    write_family_tikz(
        bsc_curves,
        out_path=tikz_dir / "fig_scaled_bsc_gexit_surface_codes_compact.tex",
        axis_name="scaledBSCGEXITSurfaceCodesCompact",
        title="Surface BSC GEXIT derivatives",
        xlabel=r"$t=h_2(p)$",
        ylabel=r"$\widetilde g_X^{\rm BSC}(t)$",
    )
    write_family_tikz(
        depolarizing_curves,
        out_path=tikz_dir / "fig_scaled_depolarizing_gexit_surface_codes_compact.tex",
        axis_name="scaledDepolarizingGEXITSurfaceCodesCompact",
        title="Surface depolarizing GEXIT derivatives",
        xlabel=r"$t=H_4(p)/2$",
        ylabel=r"$\widetilde g^{\rm depol}(t)$",
    )
    write_three_panel_figure(tikz_dir / "fig_scaled_gexit_three_panel.tex")
    write_scalings(bsc_curves, out_dir / "scaled_bsc_gexit_surface_codes_scalings.csv")
    write_scalings(
        depolarizing_curves,
        out_dir / "scaled_depolarizing_gexit_surface_codes_scalings.csv",
    )
    for path in (
        out_dir / "scaled_bsc_gexit_surface_codes_compact.png",
        out_dir / "scaled_depolarizing_gexit_surface_codes_compact.png",
        tikz_dir / "fig_scaled_bsc_gexit_surface_codes_compact.tex",
        tikz_dir / "fig_scaled_depolarizing_gexit_surface_codes_compact.tex",
        tikz_dir / "fig_scaled_gexit_three_panel.tex",
        out_dir / "scaled_bsc_gexit_surface_codes_scalings.csv",
        out_dir / "scaled_depolarizing_gexit_surface_codes_scalings.csv",
    ):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
