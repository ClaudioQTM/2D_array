#!/bin/bash

#PBS -l ncpus=32
#PBS -l mem=64GB
#PBS -l jobfs=200GB
#PBS -q normal
#PBS -P kx21
#PBS -l walltime=08:00:00
#PBS -l storage=gdata/kx21+scratch/kx21
#PBS -l wd
#PBS -r y
#PBS -J 1-5

set -euo pipefail

module load python3/3.12.1
export PATH="${HOME}/.local/bin:${PATH}"

# Run from submission directory so relative paths in main.py work.
cd "${PBS_O_WORKDIR:-$PWD}"
LOG_DIR="/g/data/kx21/${USER}/job_logs"
mkdir -p "${LOG_DIR}"

# Avoid nested threading when using joblib workers.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

OFFSETS=(10 20 30 40 50)
ARRAY_INDEX="${PBS_ARRAY_INDEX:-0}"

if [[ "${ARRAY_INDEX}" -lt 1 || "${ARRAY_INDEX}" -gt 5 ]]; then
  echo "Invalid or missing PBS_ARRAY_INDEX='${ARRAY_INDEX}'. Expected 1-5."
  exit 1
fi

OFFSET="${OFFSETS[$((ARRAY_INDEX - 1))]}"
echo "Job ${PBS_JOBID} array_index=${ARRAY_INDEX} offset=${OFFSET}"

uv run main.py "${PBS_NCPUS}" "${OFFSET}" > "${LOG_DIR}/${PBS_JOBID}_offset${OFFSET}.log" 2>&1
