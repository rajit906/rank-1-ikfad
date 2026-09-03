# Choose GPU
export CUDA_VISIBLE_DEVICES=3

# Log file name
LOGFILE="logs_besthparams/sweep_rikfad0.log"

# Run the job detached with nohup
nohup python3 -u best_hparams/sweep_rikfad0.py > "$LOGFILE" 2>&1 &

# Print info
echo "Logging to: $LOGFILE"
echo "PID: $!"
