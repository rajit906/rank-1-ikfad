#!/bin/bash

mkdir -p logs

LOGFILE="logs/sweep_rikfad0.log"

CUDA_VISIBLE_DEVICES=6 nohup python3 -u sweeps/sweep_rikfad0.py > "$LOGFILE" 2>&1 &

echo "rikfad0 sweep launched on GPU 6."
echo "Logging to: $LOGFILE"
echo "PID: $!"
