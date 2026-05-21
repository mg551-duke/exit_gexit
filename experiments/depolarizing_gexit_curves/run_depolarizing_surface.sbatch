#!/usr/bin/env bash
#SBATCH --job-name=depol-gexit
#SBATCH --output=depol-gexit-%x-%j.out
#SBATCH --error=depol-gexit-%x-%j.err
#SBATCH --time=09:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: sbatch [resource overrides] experiments/depolarizing_gexit_curves/run_depolarizing_surface.sbatch experiments/depolarizing_gexit_curves/surface_depolarizing_d7.py" >&2
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
