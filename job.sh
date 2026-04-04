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
#PBS -m e
#PBS -M yw8652@uni.sydney.edu.au

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
OFFSET="${OFFSET:-20}"

echo "Job ${PBS_JOBID} offset=${OFFSET}"
uv run main.py "${PBS_NCPUS}" "${OFFSET}" > "${LOG_DIR}/${PBS_JOBID}_offset${OFFSET}.log" 2>&1
