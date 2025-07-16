#!/bin/bash
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -N jupyterlab
#PBS -o /work/gj26/b20109/akorn/jupyterlab.out
#PBS -e /work/gj26/b20109/akorn/jupyterlab.out
#PBS -j oe
#PBS -W group_list=gj26

# 変数展開しないので以下はdepreciateされた
# PBS -o /work/gj26/b20109/akorn/logs/jupyterlab.o%J
# PBS -e /work/gj26/b20109/akorn/logs/jupyterlab.e%J

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
  jupyter lab \
  --ip=0.0.0.0 \
  --port=$PORT \
  --ServerApp.port_retries=0 \
  # ← 11007 が空いてなければ即エラー
  --ServerApp.allow_origin='*' \
  # VS Code の Origin を許可:contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}
  --ServerApp.disable_check_xsrf=True \
  # API 呼び出しの XSRF チェックをオフ:contentReference[oaicite:4]{index=4}
  --no-browser

echo "JupyterLab has been activated..."