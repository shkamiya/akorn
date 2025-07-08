# submit_all.sh
mkdir -p logs

for i in $(seq 0 17); do
  qsub -v INDEX=$i run_single_index.pbs
done