from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = ROOT / "experiments"
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from bsc_gexit_surface_sampled import write_outputs
from merge_bsc_gexit_repeats import load_result, merge_result_dicts
from surface_gexit_job_common import DEFAULT_OUT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge sharded surface BSC GEXIT jobs.")
    parser.add_argument("--distance", type=int, required=True)
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=None,
        help="Directory containing shard_* subdirectories. Defaults to the latest matching shard group.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--include-existing",
        action="store_true",
        help="Merge shards into the existing main result instead of using the first shard as the base.",
    )
    return parser.parse_args()


def latest_shard_dir(distance: int) -> Path:
    root = DEFAULT_OUT_DIR / "shards" / f"surface{distance}"
    candidates = [path for path in root.iterdir() if path.is_dir()] if root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"no shard groups found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def shard_jsons(distance: int, shard_dir: Path) -> list[Path]:
    paths = sorted(shard_dir.glob(f"shard_*/surface{distance}_bsc_gexit_sampled.json"))
    if not paths:
        raise FileNotFoundError(f"no surface{distance} shard JSON files found under {shard_dir}")
    return paths


def main() -> None:
    args = parse_args()
    shard_dir = args.shard_dir if args.shard_dir is not None else latest_shard_dir(args.distance)
    paths = shard_jsons(args.distance, shard_dir)

    if args.include_existing:
        base_path = args.out_dir / f"surface{args.distance}_bsc_gexit_sampled.json"
        if not base_path.exists():
            raise FileNotFoundError(f"existing result not found: {base_path}")
        merged = load_result(base_path)
    else:
        base_path = paths[0]
        merged = load_result(base_path)
        paths = paths[1:]

    for repeat_path in paths:
        merged = merge_result_dicts(
            merged,
            load_result(repeat_path),
            base_path=base_path,
            repeat_path=repeat_path,
        )
        base_path = None

    write_outputs(merged, args.out_dir, args.out_dir / "tikz")
    print(f"merged {len(shard_jsons(args.distance, shard_dir))} shards from {shard_dir}")


if __name__ == "__main__":
    main()
