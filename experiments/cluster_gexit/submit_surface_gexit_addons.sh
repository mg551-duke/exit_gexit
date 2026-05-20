#!/usr/bin/env bash
set -euo pipefail

SBATCH_SCRIPT="experiments/cluster_gexit/run_surface_gexit.sbatch"

submit_one() {
  local distance="$1"
  local cpus="$2"
  local mem="$3"
  local script="experiments/cluster_gexit/surface_gexit_d${distance}_addon.py"

  sbatch \
    --job-name="surface-gexit-d${distance}-addon" \
    --cpus-per-task="${cpus}" \
    --mem="${mem}" \
    "${SBATCH_SCRIPT}" \
    "${script}"
}

submit_one 7 20 64G
submit_one 11 20 64G
