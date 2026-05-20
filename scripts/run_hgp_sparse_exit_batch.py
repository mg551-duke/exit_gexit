from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exit_curve_experiments import (
    code_length_from_npz,
    plot_hybrid_paired_three_panel,
    plot_paired_derivative_result,
    plot_result,
    representative_runs,
    run_exact_class_target_code,
    run_paired_exact_class_target_code,
    write_outputs,
    write_paired_derivative_outputs,
)


CENTER_PS = [
    0.4,
    0.45,
    0.47,
    0.485,
    0.4925,
    0.495,
    0.4975,
    0.5,
    0.5025,
    0.505,
    0.5075,
    0.515,
    0.53,
    0.55,
    0.6,
]

BIT_PS = sorted(
    {
        0.0,
        0.1,
        0.2,
        0.3,
        0.35,
        0.375,
        *CENTER_PS,
        0.625,
        0.65,
        0.7,
        0.8,
        0.9,
        1.0,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sparse HGP exact-class EXIT curves with hybrid plots."
    )
    parser.add_argument("codes", type=Path, nargs="+", help="Sparse HGP .npz files")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/experiments/exit_curves"),
    )
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--component",
        choices=["x", "z", "both"],
        default="x",
        help="Exact class component to estimate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for code_path in sorted(args.codes, key=code_length_from_npz):
        n = code_length_from_npz(code_path)
        runs = args.runs if args.runs is not None else representative_runs(n)
        print(
            f"RUN {code_path.name} n={n} runs={runs} "
            f"bit_points={len(BIT_PS)} paired_points={len(CENTER_PS)}",
            flush=True,
        )

        bit_result = run_exact_class_target_code(
            code_path,
            BIT_PS,
            runs,
            seed=args.seed,
            component=args.component,
        )
        write_outputs(bit_result, args.out_dir)
        bit_png = plot_result(bit_result, args.out_dir)

        paired_result = run_paired_exact_class_target_code(
            code_path,
            CENTER_PS,
            runs,
            seed=args.seed,
            component=args.component,
        )
        write_paired_derivative_outputs(paired_result, args.out_dir)

        paired_png = plot_paired_derivative_result(paired_result, args.out_dir)
        hybrid_png = plot_hybrid_paired_three_panel(
            bit_result,
            paired_result,
            args.out_dir,
        )
        print(
            f"DONE {code_path.name} "
            f"bit_elapsed={bit_result['elapsed_seconds']:.3f} "
            f"paired_elapsed={paired_result['elapsed_seconds']:.3f}",
            flush=True,
        )
        print(f"PLOTS {bit_png} {paired_png} {hybrid_png}", flush=True)


if __name__ == "__main__":
    main()
