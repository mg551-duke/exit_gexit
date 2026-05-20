from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from bsc_gexit_surface_sampled import compute_result, entropy_centered_bsc_ps, write_outputs


DEFAULT_OUT_DIR = ROOT / "data" / "experiments" / "gexit_curves" / "entropy_centered_surface_jobs"


def ensure_surface_code(distance: int) -> Path:
    code_path = ROOT / "codes" / f"surface{distance}_HxHzLxLz.npz"
    if code_path.exists():
        return code_path

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_surface_code.py"),
            str(distance),
        ],
        cwd=ROOT,
        check=True,
    )
    return code_path


def run_surface_distance(
    distance: int,
    *,
    samples: int,
    seed: int,
    workers: int | None = None,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> None:
    if workers is None:
        workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    code_path = ensure_surface_code(distance)
    p_grid = np.array(
        entropy_centered_bsc_ps(
            edge_step=0.1,
            shoulder_step=0.05,
            center_step=0.02,
        ),
        dtype=float,
    )
    result = compute_result(
        code_path,
        samples=samples,
        seed=seed,
        p_grid=p_grid,
        workers=workers,
    )
    result["job"] = {
        "kind": "entropy-centered BSC GEXIT surface run",
        "distance": distance,
        "samples": samples,
        "seed": seed,
        "workers": workers,
        "entropy_grid": result["grid"]["t"],
    }
    write_outputs(result, out_dir, out_dir / "tikz")
