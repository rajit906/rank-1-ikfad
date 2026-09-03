#!/bin/bash

mkdir -p logs

LOGFILE="logs/sweep_adafactor.log"

CUDA_VISIBLE_DEVICES=7 nohup python3 -u sweeps/sweep_adafactor.py > "$LOGFILE" 2>&1 &

echo "adafactor sweep launched on GPU 7."
echo "Logging to: $LOGFILE"
echo "PID: $!"
