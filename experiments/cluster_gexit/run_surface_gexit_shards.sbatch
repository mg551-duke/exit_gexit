#!/usr/bin/env bash
#SBATCH --job-name=surface-gexit-shard
#SBATCH --output=surface-gexit-shard-%A_%a.out
#SBATCH --error=surface-gexit-shard-%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

if [[ -d ".venv" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

: "${SURFACE_GEXIT_DISTANCE:?set SURFACE_GEXIT_DISTANCE}"
: "${SURFACE_GEXIT_SAMPLES:?set SURFACE_GEXIT_SAMPLES}"

echo "running surface d=${SURFACE_GEXIT_DISTANCE} shard ${SLURM_ARRAY_TASK_ID:-0} with ${SURFACE_GEXIT_SAMPLES} samples"
"$PYTHON_BIN" experiments/cluster_gexit/surface_gexit_shard.py
