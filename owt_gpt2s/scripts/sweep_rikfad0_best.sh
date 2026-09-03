#!/bin/bash
# Runs the best-hyperparameter seeds for rikfad0.
# Fill in best_hparams/sweep_rikfad0.py with sweep results before running.

mkdir -p logs_besthparams

LOGFILE="logs_besthparams/sweep_rikfad0.log"

CUDA_VISIBLE_DEVICES=7 nohup python3 -u best_hparams/sweep_rikfad0.py > "$LOGFILE" 2>&1 &

echo "rikfad0 best-hparam run launched on GPU 7."
echo "Logging to: $LOGFILE"
echo "PID: $!"
