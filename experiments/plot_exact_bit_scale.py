from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = (
    ROOT
    / "data"
    / "experiments"
    / "exit_curves"
    / "surface5_HxHzLxLz_exit_rule1.json"
)
DEFAULT_TIKZ_DIR = ROOT / "data" / "experiments" / "exit_curves" / "tikz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the exact bit-scale CSS correction-entropy decomposition."
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=DEFAULT_RESULT,
        help="EXIT result JSON file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path. Defaults next to --result.",
    )
    parser.add_argument(
        "--tikz-out",
        type=Path,
        default=DEFAULT_TIKZ_DIR / "fig_surface5_bit_scale_compact.tex",
        help="Output compact PGFPlots/TikZ path.",
    )
    return parser.parse_args()


def load_result(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def plot_exact_bit_scale(result: dict, out_path: Path) -> None:
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
        values = np.array([point[key] for point in points], dtype=float)
        ax.plot(p, values, marker=".", linewidth=1.45, label=label)

    ax.axhline(
        code["k"],
        color="black",
        linestyle="--",
        linewidth=0.9,
        label=r"$k$",
    )
    ax.set_xlabel(r"erasure probability $e$")
    ax.set_ylabel("expected entropy (bits)")
    ax.set_title(rf"{code['name'].replace('_HxHzLxLz', '')}: bit scale")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def tikz_coordinates(points: list[dict], key: str) -> str:
    return " ".join(f"({point['p']:.12g},{point[key]:.12g})" for point in points)


def write_compact_tikz(result: dict, out_path: Path) -> None:
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
        "% Compact PGFPlots panel for the surface5 bit-scale decomposition.",
        "% Requires \\usepackage{pgfplots} and \\pgfplotsset{compat=1.18}.",
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        "  name=surfaceFiveBitScaleAxis,",
        r"  width=\linewidth,",
        r"  height=0.78\linewidth,",
        r"  title={Bit-scale decomposition},",
        r"  title style={font=\scriptsize},",
        r"  xlabel={$e$},",
        r"  ylabel={bits},",
        "  xmin=0, xmax=1,",
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
        rf"\addplot[black!70, dashed, line width=0.55pt] coordinates {{(0,{code['k']:.12g}) (1,{code['k']:.12g})}};",
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


def main() -> None:
    args = parse_args()
    result_path = args.result.resolve()
    out_path = (
        args.out.resolve()
        if args.out is not None
        else result_path.with_name(f"{result_path.stem}_bit_scale.png")
    )
    plot_exact_bit_scale(load_result(result_path), out_path)
    print(f"wrote {out_path}")
    tikz_path = args.tikz_out.resolve()
    write_compact_tikz(load_result(result_path), tikz_path)
    print(f"wrote {tikz_path}")


if __name__ == "__main__":
    main()
