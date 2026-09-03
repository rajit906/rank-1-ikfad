#!/bin/bash

mkdir -p logs

export CUDA_VISIBLE_DEVICES=4

nohup python3 sweep.py \
    --optimizer adafactor \
    --n_trials 100 \
    --epochs 25 \
    --device cuda \
    > logs/adafactor.log 2>&1 &

echo "Adafactor sweep started. Logging to logs/adafactor.log"
echo "PID: $!"
