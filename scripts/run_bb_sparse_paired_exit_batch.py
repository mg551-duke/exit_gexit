from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exit_curve_experiments import (  # noqa: E402
    plot_paired_derivative_result,
    run_paired_exact_class_target_code,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired exact-class EXIT curves for sparse BB codes."
    )
    parser.add_argument("codes", type=Path, nargs="+")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "experiments" / "exit_curves",
    )
    parser.add_argument("--runs", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--component", choices=["x", "z", "both"], default="x")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for code_path in args.codes:
        print(
            f"RUN {code_path.name} runs={args.runs} points={len(CENTER_PS)}",
            flush=True,
        )
        result = run_paired_exact_class_target_code(
            code_path,
            CENTER_PS,
            args.runs,
            seed=args.seed,
            component=args.component,
        )
        json_path, csv_path = write_paired_derivative_outputs(result, args.out_dir)
        png_path = plot_paired_derivative_result(result, args.out_dir)
        area = result["areas"]["exact_x_class_component_norm_dp"][
            "trapezoid_derivative_area"
        ]
        print(
            f"DONE {code_path.name} elapsed={result['elapsed_seconds']:.3f} "
            f"n={result['code']['n']} k={result['code']['k']} area={area:.10g} "
            f"files={json_path.name},{csv_path.name},{png_path.name}",
            flush=True,
        )


if __name__ == "__main__":
    main()
