#!/bin/bash
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -N eval_sudoku_akorn_sweep
#PBS -o logs/
#PBS -e logs/
#PBS -j oe
#PBS -W group_list=gj26

module purge
module load singularity

cd "$PBS_O_WORKDIR"
mkdir -p logs

# ==== 入力 ====
# qsub -v CONFIG=eval_sudoku_sweep.yaml で渡す。デフォルトは eval_sudoku_sweep.yaml
CONFIG="${CONFIG:-eval_sudoku_sweep.yaml}"
# 複数ノード分散したいときのシャーディング（直積の行を row_mod/row_rem で分割）
ROW_MOD="${ROW_MOD:-1}"
ROW_REM="${ROW_REM:-0}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

# ==== W&B ====
# export WANDB_ENTITY="${WANDB_ENTITY:-shunsuke-kamiya-the-university-of-tokyo}"
# export WANDB_PROJECT="${WANDB_PROJECT:-sudoku_eval_sweep}"
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export WANDB_API_KEY=ac9bc3f259163957d95686abca5fb49df1713b65
export WANDB_PROJECT=eval_sudoku_akorn_sweep

export PBS_JOBID="${PBS_JOBID}"

# ==== Singularity 実行 ====

singularity exec --nv \
  --bind $(pwd):/workspace \
  --bind /etc/pki/tls/certs/ca-bundle.crt:/etc/pki/tls/certs/ca-bundle.crt \
  ~/singularity/kamiya_miyabi.sif \
  bash -lc "
    cd '$PWD' && \
    python scripts/eval_sudoku_sweep.py \
      --config '$CONFIG' \
      $EXTRA_FLAGS
  "
