#!/bin/bash

# Ensure the logs directory exists
mkdir -p logs

# Set the specific GPU visibility
export CUDA_VISIBLE_DEVICES=4

nohup python3 sweep.py \
    --optimizer rikfad0 \
    --n_trials 100 \
    --epochs 25 \
    --device cuda \
    > logs/rikfad0.log 2>&1 &

echo "RIKFAD0 Sweep started. Logging to logs/rikfad0.log"
echo "PID: $!"
