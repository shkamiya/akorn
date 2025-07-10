for i in $(seq 0 2); do
  qsub -v INDEX=$i job_scripts/cifar10_myakorn_classification_sweep.pbs
done