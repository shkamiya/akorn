#!/bin/bash
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -N jupyterlab
#PBS -j oe
#PBS -W group_list=gj26

# 設定
PORT=11007
SIF=~/singularity/kamiya_miyabi.sif
WORKDIR=/work/gj26/b20109/akorn

module purge
module load singularity
cd "$WORKDIR"

# JupyterLab を Singularity 内で起動
singularity exec --nv \
  --bind $(pwd):/workspace \
  --pwd /workspace \
  "$SIF" \
  jupyter lab --ip=0.0.0.0 --port=$PORT --no-browser
