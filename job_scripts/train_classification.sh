#!/bin/bash
#PJM -L rscgrp=debug-a
#PJM -L node=1
#PJM -L elapse=00:30:00
#PJM -L jobenv=singularity
#PJM -g gb20
#PJM -j

# モジュールロード
module load singularity

# 作業ディレクトリに移動
cd /work/gb20/b20109/akorn  # 実際のパスに合わせて変更

# SingularityでPythonスクリプト実行
singularity exec --nv --bind $(pwd):/workspace singularity/akorn_wisteria.sif \
  python train_classification.py cifar10_test \
    --data cifar10 \
    --epochs 100 \
    --batchsize 64 \
    --n 2 \
    --ch 32 \
    --L 4 \
    --T 8
