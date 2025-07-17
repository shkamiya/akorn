for i in $(seq 0 9); do
  qsub -v INDEX=$i job_scripts/cifar10_akorn_resnet_slim_classification_sweep.pbs
done