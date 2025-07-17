for i in $(seq 0 23); do
  qsub -v INDEX=$i job_scripts/cifar10_myakorn_classification_limitbp_sweep.pbs
done