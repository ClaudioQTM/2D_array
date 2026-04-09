#!/bin/bash

#PBS -l ncpus=8
#PBS -l mem=16GB
#PBS -l jobfs=200GB
#PBS -q normal
#PBS -P kx21
#PBS -l walltime=08:00:00
#PBS -l storage=gdata/kx21+scratch/kx21
#PBS -l wd
#PBS -r y
#PBS -m ae
#PBS -M claudioymw@gmail.com

set -euo pipefail

module load python3/3.12.1
export PATH="${HOME}/.local/bin:${PATH}"

cd "${PBS_O_WORKDIR:-$PWD}"
LOG_DIR="/g/data/kx21/${USER}/job_logs"
mkdir -p "${LOG_DIR}"

# Avoid nested threading when using joblib workers.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Override when submitting: qsub -v OFFSET=30 job.sh
OFFSET="${OFFSET:-0}"

echo "Job ${PBS_JOBID} offset=${OFFSET}"
uv run main.py "${PBS_NCPUS}" "${OFFSET}" > "${LOG_DIR}/${PBS_JOBID}_offset${OFFSET}.log" 2>&1
