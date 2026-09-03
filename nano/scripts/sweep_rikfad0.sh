# Log file name
LOGFILE="logs/sweep_rikfad0.log"

# Run the job detached with nohup
CUDA_VISIBLE_DEVICES=4 nohup python3 -u sweeps/sweep_rikfad0.py > "$LOGFILE" 2>&1 &

# Print info
echo "Job launched on GPU 5."
echo "Logging to: $LOGFILE"
echo "PID: $!"
