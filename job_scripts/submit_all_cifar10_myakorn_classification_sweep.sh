for i in $(seq 0 0); do
  qsub -v INDEX=$i job_scripts/cifar10_akorn_resnet_slim_classification_L1T0_sweep.pbs
done