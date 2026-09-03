#!/bin/bash
# Runs the best-hyperparameter seeds for Adafactor.
# Fill in best_hparams/sweep_adafactor.py with sweep results before running.

mkdir -p logs_besthparams

LOGFILE="logs_besthparams/sweep_adafactor.log"

CUDA_VISIBLE_DEVICES=5 nohup python3 -u best_hparams/sweep_adafactor.py > "$LOGFILE" 2>&1 &

echo "Adafactor best-hparam run launched on GPU 5."
echo "Logging to: $LOGFILE"
echo "PID: $!"
