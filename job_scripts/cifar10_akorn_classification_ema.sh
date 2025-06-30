#!/bin/bash
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -N akorn_cifar10_ema
#PBS -j oe
#PBS -W group_list=gj26

module purge
module load singularity

# --- ログは $PBS_O_WORKDIR に出る ---
cd $PBS_O_WORKDIR

export WANDB_API_KEY=ac9bc3f259163957d95686abca5fb49df1713b65
export WANDB_PROJECT=akorn-cifar10_ema

TODAY=$(date '+%Y%m%d')

# --- 実行 ---
singularity exec --nv \
  --bind $(pwd):/workspace \
  ~/singularity/kamiya_miyabi.sif \
  python cifar10_akorn_classification_ema.py \
      --save-dir "results/${TODAY}_${PBS_JOBID}" \
      --wandb-project akorn_cifar10_ema  \
      --experiment-name akorn_cifar10_ema \
      --epochs 100 \
      --batch-size 64 \
      --lr 3e-4 \
      --n 2 --ch 128 --L 3 --T 3
