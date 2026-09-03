import os
import re
import pandas as pd
import numpy as np

# --- Configuration ---
OPTIMIZERS = ['adam', 'adafactor', 'ikfad', 'ikfad0', 'rikfad', 'rikfad0']
# Handle directory name differences between nano/ and owt/
POSSIBLE_LOG_DIRS = ['logs_besthparams', 'logs_best_hparams']
OUTPUT_DIR = 'sweep-table'

def get_log_dir():
    """Finds the existing log directory in the current path."""
    for d in POSSIBLE_LOG_DIRS:
        if os.path.exists(d):
            return d
    return None

def parse_gpt2_log(filepath):
    """Parses a single GPT2 log file for steps, train loss, and val loss."""
    trials = {}
    current_trial_idx = -1
    
    # Regex patterns matching the original visualization script
    trial_header_pattern = re.compile(r"--- .*? Trial (\d+) ---")
    step_pattern = re.compile(r"step\s+(\d+):\s+val\s+loss\s+([\d\.]+),\s+train\s+loss\s+([\d\.]+)")

    if not os.path.exists(filepath):
        print(f"  [!] File not found: {filepath}")
        return {}

    with open(filepath, 'r') as f:
        for line in f:
            header_match = trial_header_pattern.search(line)
            if header_match:
                current_trial_idx = int(header_match.group(1))
                # Only keep first 3 trials as per original logic limit (if applicable)
                if current_trial_idx > 2: 
                    break 
                trials[current_trial_idx] = {'step': [], 'train': [], 'val': []}
                continue

            if current_trial_idx in trials:
                step_match = step_pattern.search(line)
                if step_match:
                    trials[current_trial_idx]['step'].append(int(step_match.group(1)))
                    trials[current_trial_idx]['val'].append(float(step_match.group(2)))
                    trials[current_trial_idx]['train'].append(float(step_match.group(3)))
    
    return trials

def main():
    log_dir = get_log_dir()
    if not log_dir:
        print(f"Error: Could not find log directory ({' or '.join(POSSIBLE_LOG_DIRS)}) in {os.getcwd()}")
        return

    print(f"Processing logs from: {log_dir}")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for opt in OPTIMIZERS:
        filename = f'sweep_{opt}.log'
        filepath = os.path.join(log_dir, filename)
        
        print(f"Parsing {opt}...")
        trials_data = parse_gpt2_log(filepath)
        
        if not trials_data:
            print(f"  -> No data found for {opt}")
            continue

        # Prepare data for DataFrame
        # Each row in the CSV represents a trial (seed)
        rows = []
        for trial_id, data in trials_data.items():
            if not data['step']: continue
            
            # Create a dictionary for this row
            # Lists must be converted to strings to be compatible with ast.literal_eval in the reader
            row = {
                'step': str(data['step']),
                'train_loss': str(data['train']),
                'valid_loss': str(data['val'])
            }
            rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            out_path = os.path.join(OUTPUT_DIR, f'{opt}.csv')
            df.to_csv(out_path, index=False)
            print(f"  -> Saved {out_path}")

if __name__ == "__main__":
    main()