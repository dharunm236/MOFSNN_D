"""
Script to log val_Metric for version_64 to wandb.
"""

import pandas as pd
import wandb

# Load the updated metrics file with val_Metric
df = pd.read_csv('/home/dharunkraja/Desktop/D/MOFSNN_D/version_64/val_metrics_with_val_Metric.csv', index_col=0)

print("Logging val_Metric to wandb for version_64...")
print(f"Best epoch: {df['val_Metric'].idxmax()}")
print(f"Best val_Metric: {df['val_Metric'].max():.6f}")

# Initialize wandb run
wandb.init(
    project="MOFSNN_D",  # Change to your project name if different
    name="version_64_val_Metric",
    config={
        "tasks": ['TSD', 'SSD', 'WS24_water', 'WS24_water4', 'WS24_acid', 'WS24_base', 'WS24_boiling'],
        "task_weights": [0.36510918, 0.24277999, 0.16388823, 0.16388823, 0.02629725, 0.01878375, 0.01925334],
        "description": "val_Metric logging for version_64"
    }
)

# Log val_Metric for each epoch
for epoch, row in df.iterrows():
    wandb.log({
        "epoch": epoch,
        "val_Metric": row['val_Metric'],
        "TSD/val_R2Score": row['TSD/val_R2Score'],
        "TSD/val_MeanAbsoluteError": row['TSD/val_MeanAbsoluteError'],
        "SSD/val_BalancedAccuracy": row['SSD/val_BalancedAccuracy'],
        "SSD/val_AUROC": row['SSD/val_AUROC'],
        "WS24_water/val_BalancedAccuracy": row['WS24_water/val_BalancedAccuracy'],
        "WS24_water/val_AUROC": row['WS24_water/val_AUROC'],
        "WS24_water4/val_BalancedAccuracy": row['WS24_water4/val_BalancedAccuracy'],
        "WS24_water4/val_AUROC": row['WS24_water4/val_AUROC'],
        "WS24_acid/val_BalancedAccuracy": row['WS24_acid/val_BalancedAccuracy'],
        "WS24_acid/val_AUROC": row['WS24_acid/val_AUROC'],
        "WS24_base/val_BalancedAccuracy": row['WS24_base/val_BalancedAccuracy'],
        "WS24_base/val_AUROC": row['WS24_base/val_AUROC'],
        "WS24_boiling/val_BalancedAccuracy": row['WS24_boiling/val_BalancedAccuracy'],
        "WS24_boiling/val_AUROC": row['WS24_boiling/val_AUROC'],
    })

# Log summary metrics
wandb.run.summary["best_val_Metric"] = df['val_Metric'].max()
wandb.run.summary["best_epoch"] = int(df['val_Metric'].idxmax())
wandb.run.summary["mean_val_Metric"] = df['val_Metric'].mean()

wandb.finish()
print("\nSuccessfully logged to wandb!")
