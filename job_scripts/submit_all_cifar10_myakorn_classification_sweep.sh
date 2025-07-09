for i in $(seq 0 17); do
  qsub -v INDEX=$i job_scripts/cifar10_myakorn_classification_sweep.pbs
done