#!/usr/bin/env bash
set -euo pipefail

SBATCH_SCRIPT="experiments/depolarizing_gexit_curves/run_depolarizing_surface.sbatch"

sbatch \
  --job-name="depol-gexit-d9" \
  --cpus-per-task=20 \
  --mem=64G \
  --time=18:00:00 \
  "${SBATCH_SCRIPT}" \
  "experiments/depolarizing_gexit_curves/surface_depolarizing_d9.py"

sbatch \
  --job-name="depol-gexit-d11" \
  --cpus-per-task=8 \
  --mem=64G \
  --time=24:00:00 \
  "${SBATCH_SCRIPT}" \
  "experiments/depolarizing_gexit_curves/surface_depolarizing_d11.py"
