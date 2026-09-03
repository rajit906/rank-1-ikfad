# Choose GPU
export CUDA_VISIBLE_DEVICES=0

# Log file name
LOGFILE="logs/sweep_rikfad.log"

# Run the job detached with nohup
nohup python3 -u sweeps/sweep_rikfad.py > "$LOGFILE" 2>&1 &

# Print info
echo "Logging to: $LOGFILE"
echo "PID: $!"
