#!/usr/bin/env bash
#SBATCH --job-name=surface-gexit
#SBATCH --output=surface-gexit-%j.out
#SBATCH --error=surface-gexit-%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: sbatch [resource overrides] experiments/cluster_gexit/run_surface_gexit.sbatch experiments/cluster_gexit/surface_gexit_d21.py" >&2
  exit 2
fi

SCRIPT="$1"
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

echo "running ${SCRIPT} with ${SLURM_CPUS_PER_TASK:-1} coupled sample-batch worker processes"
"$PYTHON_BIN" "$SCRIPT"
