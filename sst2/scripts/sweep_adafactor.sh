#!/bin/bash

mkdir -p logs

export CUDA_VISIBLE_DEVICES=1

nohup python3 sweep.py \
    --optimizer adafactor \
    --n_trials 80 \
    --epochs 2 \
    --device cuda \
    > logs/adafactor.log 2>&1 &

echo "adafactor sweep started on GPU 1. Logging to logs/adafactor.log"
echo "PID: $!"
