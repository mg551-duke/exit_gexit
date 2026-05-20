from __future__ import annotations

import os
from pathlib import Path

from surface_gexit_job_common import DEFAULT_OUT_DIR, run_surface_distance


def env_int(name: str, default: int | None = None) -> int:
    value = os.environ.get(name)
    if value is None:
        if default is None:
            raise ValueError(f"missing required environment variable {name}")
        return default
    return int(value)


if __name__ == "__main__":
    distance = env_int("SURFACE_GEXIT_DISTANCE")
    samples = env_int("SURFACE_GEXIT_SAMPLES")
    seed = env_int("SURFACE_GEXIT_SEED")
    workers = env_int("SURFACE_GEXIT_WORKERS", env_int("SLURM_CPUS_PER_TASK", 1))
    repeat_label = os.environ.get(
        "SURFACE_GEXIT_REPEAT_LABEL",
        f"surface{distance}_addon_{samples}_seed{seed}",
    )
    out_dir = Path(os.environ.get("SURFACE_GEXIT_OUT_DIR", str(DEFAULT_OUT_DIR)))

    run_surface_distance(
        distance=distance,
        samples=samples,
        seed=seed,
        workers=workers,
        out_dir=out_dir,
        merge_existing=True,
        repeat_label=repeat_label,
    )
