for i in $(seq 0 11); do
  qsub -v INDEX=$i job_scripts/cifar10_myakorn_small_kernel_sweep.pbs
done