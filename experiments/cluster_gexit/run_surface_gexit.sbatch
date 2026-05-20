#!/usr/bin/env bash
#SBATCH --job-name=surface-gexit
#SBATCH --array=0-4
#SBATCH --output=surface-gexit-%A_%a.out
#SBATCH --error=surface-gexit-%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G

set -euo pipefail

SCRIPTS=(
  "experiments/cluster_gexit/surface_gexit_d7.py"
  "experiments/cluster_gexit/surface_gexit_d11.py"
  "experiments/cluster_gexit/surface_gexit_d15.py"
  "experiments/cluster_gexit/surface_gexit_d21.py"
  "experiments/cluster_gexit/surface_gexit_d45.py"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SCRIPT="${SCRIPTS[$TASK_ID]}"
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

echo "running ${SCRIPT} with ${SLURM_CPUS_PER_TASK:-1} worker processes"
"$PYTHON_BIN" "$SCRIPT"
