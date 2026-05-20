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
    seed_base = env_int("SURFACE_GEXIT_SEED_BASE", distance * 1_000_000)
    shard_index = env_int(
        "SURFACE_GEXIT_SHARD_INDEX",
        env_int("SLURM_ARRAY_TASK_ID", 0),
    )
    shard_count = env_int("SURFACE_GEXIT_SHARD_COUNT", env_int("SLURM_ARRAY_TASK_COUNT", 1))

    output_root = Path(os.environ.get("SURFACE_GEXIT_OUT_DIR", str(DEFAULT_OUT_DIR)))
    out_dir = (
        output_root
        / "shards"
        / f"surface{distance}"
        / f"samples{samples}_seed{seed_base}_count{shard_count}"
        / f"shard_{shard_index:03d}"
    )
    print(
        f"surface d={distance} shard {shard_index + 1}/{shard_count}: "
        f"{samples} samples, seed={seed_base + shard_index}, out={out_dir}",
        flush=True,
    )
    run_surface_distance(
        distance=distance,
        samples=samples,
        seed=seed_base + shard_index,
        workers=1,
        out_dir=Path(out_dir),
    )
