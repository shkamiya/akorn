for i in $(seq 0 1); do
  qsub -v INDEX=$i job_scripts/cifar10_akorn_resnet_slim_classification_L0_sweep.pbs
done