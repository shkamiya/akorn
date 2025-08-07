#!/bin/bash

# Instructions for running the Wisteria sweep job:
# 1. First, create a wandb sweep on Wisteria:
#    singularity exec kamiya_miyabi.sif python -m wandb sweep configs/_0801_2025_sweep_sudoku.yaml
#
# 2. Save the returned sweep ID:
#    echo "YOUR_SWEEP_ID_HERE" > sweep_id.txt
#
# 3. Submit the job:
#    pjsub job_scripts/train_sudoku_akorn_sweep.pjm
#
# 4. OR submit with specific sweep ID as environment variable:
#    pjsub -x WANDB_SWEEP_ID=your_sweep_id job_scripts/train_sudoku_akorn_sweep.pjm
#
# 5. For multiple parallel agents:
#    for i in {1..4}; do
#        pjsub job_scripts/train_sudoku_akorn_sweep.pjm
#    done

echo "Wisteria Sweep Job Setup Complete!"
echo ""
echo "To run on Wisteria:"
echo "1. Create sweep: singularity exec kamiya_miyabi.sif python -m wandb sweep configs/_0801_2025_sweep_sudoku.yaml"
echo "2. Save sweep ID: echo 'SWEEP_ID' > sweep_id.txt"
echo "3. Submit job: pjsub job_scripts/train_sudoku_akorn_sweep.pjm"
echo ""
echo "Files ready:"
echo "- job_scripts/train_sudoku_akorn_sweep.pjm"
echo "- configs/_0801_2025_sweep_sudoku.yaml"
echo "- sweep_id.txt (test file - replace with actual sweep ID)"