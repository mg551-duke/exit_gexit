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
DEFAULT_CLUSTER_DIR = DEFAULT_GEXIT_DIR / "entropy_centered_surface_jobs"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay entropy-axis BSC GEXIT curves for surface codes."
    )
    parser.add_argument(
        "--gexit-dir",
        type=Path,
        default=DEFAULT_CLUSTER_DIR,
        help="Directory containing surface*_bsc_gexit_sampled.csv files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to --gexit-dir.",
    )
    parser.add_argument(
        "--include-surface5",
        dest="include_surface5",
        action="store_true",
        default=True,
        help="Include the exact surface5 curve from the parent gexit_curves directory.",
    )
    parser.add_argument(
        "--no-include-surface5",
        dest="include_surface5",
        action="store_false",
    )
    return parser.parse_args()


def binary_entropy(prob: float) -> float:
    if prob <= 0.0:
        return 0.0
    if prob >= 0.5:
        return 1.0
    return -prob * math.log2(prob) - (1.0 - prob) * math.log2(1.0 - prob)


def binary_entropy_derivative(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    derivative = np.full_like(p, np.nan)
    derivative[p == 0.0] = np.inf
    mask = (p > 0.0) & (p < 0.5)
    derivative[mask] = np.log2((1.0 - p[mask]) / p[mask])
    return derivative


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_metadata(csv_path: Path) -> dict:
    with csv_path.with_suffix(".json").open(encoding="utf-8") as f:
        return json.load(f)["code"]


def surface_distance(path: Path, metadata: dict) -> int:
    if "distance" in metadata and metadata["distance"] is not None:
        return int(metadata["distance"])
    match = re.search(r"surface(\d+)", path.name)
    if match is None:
        raise ValueError(f"could not infer surface distance from {path.name}")
    return int(match.group(1))


def derivative_key(rows: list[dict[str, str]], csv_path: Path) -> str:
    if "posterior_x_class_component_norm_dp" in rows[0]:
        return "posterior_x_class_component_norm_dp"
    if "exact_x_class_component_norm_dp" in rows[0]:
        return "exact_x_class_component_norm_dp"
    raise ValueError(f"{csv_path} has no supported GEXIT derivative column")


def load_curve(csv_path: Path) -> Curve:
    rows = read_rows(csv_path)
    if not rows:
        raise ValueError(f"{csv_path} is empty")
    metadata = read_metadata(csv_path)
    key = derivative_key(rows, csv_path)
    p = np.array([float(row["p"]) for row in rows], dtype=float)
    dydp = np.array([float(row[key]) for row in rows], dtype=float)
    order = np.argsort(p)
    p = p[order]
    dydp = dydp[order]

    t = np.array([binary_entropy(float(value)) for value in p], dtype=float)
    hprime = binary_entropy_derivative(p)
    dydt = np.divide(
        dydp,
        hprime,
        out=np.full_like(dydp, np.nan),
        where=hprime > 0.0,
    )
    finite = np.isfinite(dydt)
    if not np.any(finite):
        raise ValueError(f"{csv_path} has no finite entropy-axis GEXIT values")
    peak = float(np.max(dydt[finite]))
    scale = 1.0 / peak if peak > 0.0 else 1.0
    area = float(np.trapezoid(dydt[finite], t[finite]))
    return Curve(
        distance=surface_distance(csv_path, metadata),
        n=int(metadata["n"]),
        k=int(metadata["k"]),
        csv_path=csv_path,
        t=t[finite],
        y=dydt[finite],
        scale=scale,
        area=area,
    )


def discover_curves(gexit_dir: Path, include_surface5: bool) -> list[Curve]:
    csv_paths = sorted(gexit_dir.glob("surface*_bsc_gexit_sampled.csv"))
    if include_surface5:
        surface5 = DEFAULT_GEXIT_DIR / "surface5_HxHzLxLz_bsc_gexit.csv"
        if surface5.exists():
            csv_paths.insert(0, surface5)
    curves = [load_curve(csv_path) for csv_path in csv_paths]
    unique: dict[int, Curve] = {}
    for curve in curves:
        unique[curve.distance] = curve
    return [unique[distance] for distance in sorted(unique)]


def format_scale(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) >= 1000.0 or abs(value) < 0.01:
        exponent = int(math.floor(math.log10(abs(value))))
        mantissa = value / (10.0**exponent)
        return rf"{mantissa:.3g}\times 10^{{{exponent}}}"
    return f"{value:.3g}"


def tikz_coordinates(curve: Curve) -> str:
    y = curve.y * curve.scale
    return " ".join(
        f"({t_value:.12g},{y_value:.12g})"
        for t_value, y_value in zip(curve.t, y)
    )


def write_tikz(curves: list[Curve], out_path: Path) -> None:
    colors = ["blue", "orange", "green!60!black", "red", "purple", "brown"]
    markers = ["*", "square*", "triangle*", "diamond*", "pentagon*", "otimes*"]
    lines = [
        "% Compact PGFPlots panel for entropy-axis BSC GEXIT surface derivatives.",
        "% Requires \\usepackage{pgfplots} and \\pgfplotsset{compat=1.18}.",
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        "  name=scaledBSCGEXITSurfaceEntropyAxisCompact,",
        r"  width=\linewidth,",
        r"  height=0.78\linewidth,",
        r"  title={Surface BSC GEXIT derivatives},",
        r"  title style={font=\scriptsize},",
        r"  xlabel={$t=h_2(p)$},",
        r"  ylabel={$\widetilde g_X^{\rm BSC}(t)$},",
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


def write_scalings(curves: list[Curve], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["distance", "n", "k", "csv", "area", "scale"],
        )
        writer.writeheader()
        for curve in curves:
            writer.writerow(
                {
                    "distance": curve.distance,
                    "n": curve.n,
                    "k": curve.k,
                    "csv": str(curve.csv_path),
                    "area": f"{curve.area:.12g}",
                    "scale": f"{curve.scale:.12g}",
                }
            )


def plot_png(curves: list[Curve], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
    ax.axvline(0.5, color="black", linestyle=":", linewidth=0.9)
    for curve in curves:
        ax.plot(
            curve.t,
            curve.y * curve.scale,
            marker=".",
            linewidth=1.45,
            label=rf"$d={curve.distance}$, scaled by ${format_scale(curve.scale)}$",
        )
    ax.set_xlabel(r"BSC channel entropy $t=h_2(p)$")
    ax.set_ylabel(r"$\widetilde g_X^{\rm BSC}(t)$")
    ax.set_title("Surface BSC GEXIT derivatives")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.02, 1.06)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.resolve() if args.out_dir is not None else args.gexit_dir.resolve()
    curves = discover_curves(args.gexit_dir.resolve(), args.include_surface5)
    plot_png(curves, out_dir / "scaled_bsc_gexit_surface_codes_entropy_axis.png")
    write_tikz(
        curves,
        out_dir / "tikz" / "fig_scaled_bsc_gexit_surface_codes_entropy_axis_compact.tex",
    )
    write_scalings(curves, out_dir / "scaled_bsc_gexit_surface_codes_entropy_axis_scalings.csv")
    print(f"wrote {out_dir / 'scaled_bsc_gexit_surface_codes_entropy_axis.png'}")
    print(
        "wrote "
        f"{out_dir / 'tikz' / 'fig_scaled_bsc_gexit_surface_codes_entropy_axis_compact.tex'}"
    )
    print(f"wrote {out_dir / 'scaled_bsc_gexit_surface_codes_entropy_axis_scalings.csv'}")


if __name__ == "__main__":
    main()
