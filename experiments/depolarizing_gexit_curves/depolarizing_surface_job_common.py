from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from depolarizing_surface_sampled import (  # noqa: E402
    compute_result,
    inverse_depolarizing_entropy,
    write_outputs,
)


DEFAULT_OUT_DIR = (
    ROOT
    / "data"
    / "experiments"
    / "depolarizing_gexit_curves"
    / "cluster_surface_jobs"
)

NORMALIZED_ENTROPY_GRID_20 = (
    0.0,
    0.1,
    0.2,
    0.3,
    0.35,
    0.37,
    0.4,
    0.43,
    0.46,
    0.49,
    0.51,
    0.54,
    0.57,
    0.6,
    0.63,
    0.65,
    0.7,
    0.8,
    0.9,
    1.0,
)


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


def depolarizing_p_grid_20() -> np.ndarray:
    return np.array(
        [inverse_depolarizing_entropy(2.0 * t) for t in NORMALIZED_ENTROPY_GRID_20],
        dtype=float,
    )


def run_depolarizing_surface_distance(
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
    p_grid = depolarizing_p_grid_20()
    result = compute_result(
        code_path,
        p_grid=p_grid,
        samples=samples,
        seed=seed,
        workers=max(1, workers),
        progress=True,
    )
    result["job"] = {
        "kind": "20-point normalized-entropy depolarizing GEXIT surface run",
        "distance": distance,
        "samples": samples,
        "seed": seed,
        "workers": max(1, workers),
        "normalized_entropy_grid": list(NORMALIZED_ENTROPY_GRID_20),
    }
    write_outputs(result, out_dir)
