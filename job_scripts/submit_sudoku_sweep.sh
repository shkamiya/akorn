for i in $(seq 0 17); do
  qsub -v INDEX=$i job_scripts/train_sudoku_akorn_sweep.sh
done