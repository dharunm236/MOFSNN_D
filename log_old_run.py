import pandas as pd
import wandb
import yaml
import os
import sys

# Path to the specific version directory
LOG_DIR = r"CGCNN_MT/logs/TSD_SSD_WS24_water_WS24_water4_WS24_acid_WS24_base_WS24_boiling_seed42_att_cgcnn/version_64"
METRICS_FILE = os.path.join(LOG_DIR, "val_metrics.csv")
HPARAMS_FILE = os.path.join(LOG_DIR, "hparams.yaml")

def load_hparams(path):
    """
    Attempts to load hparams.yaml.
    Since the file contains python-specific tags (!!python/...), safe_load will fail.
    We try to load it, but if it fails due to missing classes, we might need a workaround 
    or just skip the complex objects.
    """
    try:
        with open(path, 'r') as f:
            # Try full load if possible, might fail if classes aren't in path
            return yaml.load(f, Loader=yaml.Loader)
    except Exception as e:
        print(f"Warning: Could not fully load hparams.yaml due to: {e}")
        print("Attempting to load as plain text and parse simple keys (fallback)...")
        
        # Fallback: simple parsing of top-level keys
        config = {}
        try:
            with open(path, 'r') as f:
                for line in f:
                    if ':' in line and not line.strip().startswith(('aa', '!!', '-')):
                        key, val = line.split(':', 1)
                        key = key.strip()
                        val = val.strip()
                        if val and not val.startswith('!!'):
                            config[key] = val
            return config
        except Exception as e2:
            print(f"Failed fallback parsing: {e2}")
            return {}

def main():
    print(f"Looking for metrics at: {METRICS_FILE}")
    if not os.path.exists(METRICS_FILE):
        print("Error: Metrics file not found!")
        return

    # User Input for WandB
    project_name = input("Enter WandB Project Name [MOFSNN_D]: ").strip() or "MOFSNN_D"
    run_name = input("Enter WandB Run Name [version_64_restored]: ").strip() or "version_64_restored"
    entity = input("Enter WandB Entity (username/team) [leave empty for default]: ").strip()
    
    if entity == "":
        entity = None

    # Load Data
    df = pd.read_csv(METRICS_FILE)
    
    # Load Config
    config = {}
    if os.path.exists(HPARAMS_FILE):
        config = load_hparams(HPARAMS_FILE)
    
    print(f"Found {len(df)} epochs of data.")
    
    # Initialize WandB
    print("Initializing WandB...")
    wandb.init(
        project=project_name,
        name=run_name,
        entity=entity,
        config=config,
        resume="allow" # Allows adding to existing runs if ID matches, though here we make a new one
    )

    print("Logging metrics...")
    # Iterate and log
    for index, row in df.iterrows():
        # The first column is usually unnamed or 'epoch' if it was index.
        # Based on file content: ",TSD/val_R2Score..." 
        # The first column in the CSV data provided earlier was the index/epoch (1, 2, 3...)
        
        # row is a Series. 
        # If the first column is the index in the CSV but not named, pandas might name it 'Unnamed: 0'
        step = index + 1 # Default step if no epoch column
        
        data_to_log = row.to_dict()
        
        # Check if there is an explicit epoch column
        possible_epoch_keys = ['epoch', 'Unnamed: 0']
        for key in possible_epoch_keys:
            if key in data_to_log:
                step = int(data_to_log[key])
                del data_to_log[key] # Remove step from metrics
                break
        
        wandb.log(data_to_log, step=step)
        
        if step % 10 == 0:
            print(f"Logged epoch {step}...")

    print("Finished logging.")
    wandb.finish()

if __name__ == "__main__":
    main()
