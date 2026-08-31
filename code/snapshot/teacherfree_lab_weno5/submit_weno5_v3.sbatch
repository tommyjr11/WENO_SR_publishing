#!/usr/bin/env bash
#SBATCH --job-name=weno5-v3-tv
#SBATCH --partition=granularmech
#SBATCH --account=comgranmech
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --output=/home/rliuca/weno5_v3_%j.out
#SBATCH --error=/home/rliuca/weno5_v3_%j.err

set -euo pipefail

cd /home/rliuca

# Respect the exact GPU IDs assigned by Slurm, including nonzero IDs or UUIDs.
IFS=',' read -r GPU_A GPU_B _ <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
if [[ -z "${GPU_B:-}" ]]; then
  echo "Slurm exposed fewer than two GPUs: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
  exit 1
fi
echo "Slurm GPUs: GPU_A=$GPU_A GPU_B=$GPU_B"

# Keep the Slurm allocation alive until both GPU training processes finish.
GPU_A="$GPU_A" GPU_B="$GPU_B" WAIT_FOR_JOBS=1 \
  bash teacherfree_lab_weno5/run_weno5_v3_tv001_002.sh
