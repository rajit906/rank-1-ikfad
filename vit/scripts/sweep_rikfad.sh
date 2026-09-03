#!/bin/bash

# Ensure the logs directory exists
mkdir -p logs

# Set the specific GPU visibility
export CUDA_VISIBLE_DEVICES=4

nohup python3 sweep.py \
    --optimizer rikfad \
    --n_trials 100 \
    --epochs 25 \
    --device cuda \
    > logs/rikfad.log 2>&1 &

echo "RIKFAD Sweep started. Logging to logs/rikfad.log"
echo "PID: $!"
