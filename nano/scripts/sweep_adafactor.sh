LOGFILE="logs/sweep_adafactor.log"

CUDA_VISIBLE_DEVICES=1 nohup python3 -u sweeps/sweep_adafactor.py > "$LOGFILE" 2>&1 &

echo "Adafactor sweep launched."
echo "Logging to: $LOGFILE"
echo "PID: $!"
