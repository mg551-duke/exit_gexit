#!/usr/bin/env bash
set -euo pipefail

SBATCH_SCRIPT="experiments/depolarizing_gexit_curves/run_depolarizing_surface.sbatch"

submit_one() {
  local distance="$1"
  local cpus="$2"
  local mem="$3"
  local time_limit="$4"
  local script="experiments/depolarizing_gexit_curves/surface_depolarizing_d${distance}.py"

  sbatch \
    --job-name="depol-gexit-d${distance}" \
    --cpus-per-task="${cpus}" \
    --mem="${mem}" \
    --time="${time_limit}" \
    "${SBATCH_SCRIPT}" \
    "${script}"
}

submit_one 5 20 64G 03:00:00
submit_one 7 20 64G 06:00:00
submit_one 9 20 64G 18:00:00
submit_one 11 8 64G 24:00:00
