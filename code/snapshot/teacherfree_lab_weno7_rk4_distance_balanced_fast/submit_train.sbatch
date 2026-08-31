#!/usr/bin/env bash
#SBATCH --job-name=w7-rk4-fast
#SBATCH --partition=granularmech
#SBATCH --account=comgranmech
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=7-00:00:00
#SBATCH --output=w7-rk4-fast-%j.out

set -euo pipefail

if [[ -f "${SLURM_SUBMIT_DIR}/run_train.sh" ]]; then
  PACKAGE_DIR="${SLURM_SUBMIT_DIR}"
elif [[ -f "${SLURM_SUBMIT_DIR}/teacherfree_lab_weno7_rk4_distance_balanced_fast/run_train.sh" ]]; then
  PACKAGE_DIR="${SLURM_SUBMIT_DIR}/teacherfree_lab_weno7_rk4_distance_balanced_fast"
else
  echo "cannot locate WENO7 package from SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}" >&2
  exit 2
fi
cd "$PACKAGE_DIR"
WAIT_FOR_TRAIN=1 GPU_ID=0 bash run_train.sh
