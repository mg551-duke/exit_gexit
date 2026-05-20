#!/usr/bin/env bash
set -euo pipefail

SBATCH_SCRIPT="experiments/cluster_gexit/run_surface_gexit_shards.sbatch"

submit_shards() {
  local distance="$1"
  local shard_count="$2"
  local samples_per_shard="$3"
  local seed_base="$4"
  local mem="$5"

  sbatch \
    --job-name="surface-gexit-d${distance}-shard" \
    --array="0-$((shard_count - 1))" \
    --cpus-per-task=1 \
    --mem="${mem}" \
    --export="ALL,SURFACE_GEXIT_DISTANCE=${distance},SURFACE_GEXIT_SAMPLES=${samples_per_shard},SURFACE_GEXIT_SEED_BASE=${seed_base},SURFACE_GEXIT_SHARD_COUNT=${shard_count}" \
    "${SBATCH_SCRIPT}"
}

# Same total sample targets as the monolithic jobs, but spread across many
# one-worker jobs to avoid multiplying memory inside a single process.
submit_shards 15 20 750 15015000 64G
submit_shards 21 20 500 21021000 64G
submit_shards 45 20 250 45045000 64G
