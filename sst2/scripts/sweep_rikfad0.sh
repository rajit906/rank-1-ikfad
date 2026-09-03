#!/bin/bash

mkdir -p logs

export CUDA_VISIBLE_DEVICES=0

nohup python3 sweep.py \
    --optimizer rikfad0 \
    --n_trials 80 \
    --epochs 2 \
    --device cuda \
    > logs/rikfad0.log 2>&1 &

echo "rikfad0 sweep started on GPU 0. Logging to logs/rikfad0.log"
echo "PID: $!"
