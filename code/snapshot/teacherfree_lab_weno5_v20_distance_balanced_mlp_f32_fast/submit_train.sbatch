#!/usr/bin/env bash
#SBATCH --job-name=w5-f32-fast
#SBATCH --partition=granularmech
#SBATCH --account=comgranmech
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=7-00:00:00
#SBATCH --output=w5-f32-fast-%j.out

set -euo pipefail

if [[ -f "${SLURM_SUBMIT_DIR}/run_train.sh" ]]; then
  PACKAGE_DIR="${SLURM_SUBMIT_DIR}"
elif [[ -f "${SLURM_SUBMIT_DIR}/teacherfree_lab_weno5_v20_distance_balanced_mlp_f32_fast/run_train.sh" ]]; then
  PACKAGE_DIR="${SLURM_SUBMIT_DIR}/teacherfree_lab_weno5_v20_distance_balanced_mlp_f32_fast"
else
  echo "cannot locate WENO5 package from SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR}" >&2
  exit 2
fi

cd "$PACKAGE_DIR"

export MODULE_NAME="${MODULE_NAME:-miniconda3/24.3.0-quc3pyu}"
export CONDA_ENV="${CONDA_ENV:-base}"
export GPU_ID=0
export WAIT_FOR_TRAIN=1

bash run_train.sh
