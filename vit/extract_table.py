import os
import re
import pandas as pd
import ast

# Configuration
log_dir = 'logs_besthparams'     # Updated to match where run_sst2.sh saves logs
output_dir = 'sweep-table'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def parse_log_file(file_path):
    """
    Parses a single log file and returns a dictionary arranged by seed.
    Structure: {seed_id: {'train_loss': [v1, v2], 'train_acc': [v1, v2], ...}}
    """
    data_by_seed = {}
    
    # Regex to capture metrics from the Step line in run_best_seeds.py
    # Matches line format: 
    # "  S0 | Ep 1 | Step 087/872 | Train Loss: 0.5512 | Train Acc: 71.20% | Test Loss: 0.4412 | Test Acc: 81.02%"
    step_pattern = re.compile(
        r"S(\d+)\s+\|\s+"                      # Capture Seed (S0)
        r"Ep\s+\d+\s+\|\s+"                    # Match Epoch (ignore value for regex)
        r"Step\s+\d+/\d+\s+\|\s+"              # Match Step info
        r"Train Loss:\s+([\d\.]+)\s+\|\s+"     # Capture Train Loss (Group 2)
        r"Train Acc:\s+([\d\.]+)\%\s+\|\s+"    # Capture Train Acc (Group 3)
        r"Test Loss:\s+([\d\.]+)\s+\|\s+"      # Capture Test Loss (Group 4)
        r"Test Acc:\s+([\d\.]+)\%"             # Capture Test Acc (Group 5)
    )

    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Filter for relevant lines starting with "S" followed by digit
                # Example: "S0 | Ep 1..."
                if not line.startswith("S") or "|" not in line:
                    continue

                match = step_pattern.search(line)
                if match:
                    seed_id = int(match.group(1))
                    
                    # Initialize dict for this seed if it doesn't exist yet
                    if seed_id not in data_by_seed:
                        data_by_seed[seed_id] = {
                            'train_loss': [],
                            'train_acc': [],
                            'valid_loss': [],
                            'valid_acc': []
                        }
                    
                    # Extract values
                    t_loss = float(match.group(2))
                    t_acc = float(match.group(3))
                    test_loss = float(match.group(4))
                    test_acc = float(match.group(5))
                    
                    # Append to lists
                    data_by_seed[seed_id]['train_loss'].append(t_loss)
                    data_by_seed[seed_id]['train_acc'].append(t_acc)
                    data_by_seed[seed_id]['valid_loss'].append(test_loss)
                    data_by_seed[seed_id]['valid_acc'].append(test_acc)
                        
        return data_by_seed
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return {}

def process_all_logs():
    """Iterates through logs directory and creates CSVs."""
    if not os.path.exists(log_dir):
        print(f"Directory '{log_dir}' does not exist.")
        return

    log_files = [f for f in os.listdir(log_dir) if f.endswith('.log') and f != 'global.log']
    
    if not log_files:
        print(f"No specific optimizer .log files found in {log_dir}")
        return

    print(f"Found {len(log_files)} log files in {log_dir}. Processing...")

    for log_file in log_files:
        optimizer_name = os.path.splitext(log_file)[0] # e.g., 'adam'
        input_path = os.path.join(log_dir, log_file)
        
        # Parse data
        parsed_data = parse_log_file(input_path)
        
        if not parsed_data:
            print(f"No data extracted for {optimizer_name} (File might be empty or format mismatch)")
            continue

        # Convert to DataFrame
        # Rows = Seeds, Columns = Lists of metric values over time
        rows = []
        for seed, metrics in parsed_data.items():
            row = {
                'seed': seed,
                'train_loss': metrics['train_loss'],
                'train_acc': metrics['train_acc'],
                'valid_loss': metrics['valid_loss'], # Replaced valid with test
                'valid_acc': metrics['valid_acc']    # Replaced valid with test
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by seed to be clean
        df = df.sort_values(by='seed')

        # Save to CSV
        output_path = os.path.join(output_dir, f"{optimizer_name}.csv")
        df.to_csv(output_path, index=False)
        print(f"Saved {output_path}")

if __name__ == "__main__":
    process_all_logs()