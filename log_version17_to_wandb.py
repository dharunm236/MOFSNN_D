"""
Script to manually log version_17 metrics to wandb.
Best model: logs/TSD_SSD_WS24_water_WS24_water4_WS24_acid_WS24_base_WS24_boiling_seed42_att_cgcnn/version_17

val_Metric is computed as a weighted sum of per-task metrics:
  - regression tasks: R2Score * task_weight
  - classification tasks: BalancedAccuracy * task_weight
  (using fixed_weight_sum aggregation)
"""

import pandas as pd
import wandb
import yaml
import os
from pathlib import Path

# Model directory
MODEL_DIR = Path("/home/dharunkraja/Desktop/D/MOFSNN_D/logs/TSD_SSD_WS24_water_WS24_water4_WS24_acid_WS24_base_WS24_boiling_seed42_att_cgcnn/version_17")
VAL_METRICS_FILE = MODEL_DIR / "val_metrics.csv"
TEST_METRICS_FILE = MODEL_DIR / "test_metrics.csv"
HPARAMS_FILE = MODEL_DIR / "hparams.yaml"

# From hparams.yaml — the exact values used during training
TASKS = ['TSD', 'SSD', 'WS24_water', 'WS24_water4', 'WS24_acid', 'WS24_base', 'WS24_boiling']
TASK_TYPES = ['regression', 'classification', 'classification', 'classification_4', 'classification', 'classification', 'classification']
TASK_WEIGHTS = [
    0.36510918055881664,
    0.242779995304062,
    0.16388823667527588,
    0.16388823667527588,
    0.026297252876262032,
    0.018783752054472882,
    0.019253345855834703,
]


def compute_val_metric(row, split='val'):
    """
    Reproduce the val_Metric calculation from CGCNN_MT/module/module.py _epoch_eval().
    For regression tasks:  metric = R2Score
    For classification tasks: metric = BalancedAccuracy
    val_Metric = sum(metric_i * task_weight_i)  (fixed_weight_sum)
    """
    monitor_metric = 0.0
    for task, task_type, weight in zip(TASKS, TASK_TYPES, TASK_WEIGHTS):
        if task_type == 'regression':
            col = f"{task}/{split}_R2Score"
        else:  # classification or classification_4
            col = f"{task}/{split}_BalancedAccuracy"
        
        if col in row.index and pd.notna(row[col]):
            monitor_metric += row[col] * weight
    return monitor_metric


def load_hparams(path):
    """Load hyperparameters from yaml file, handling python-specific tags."""
    config = {}
    try:
        with open(path, 'r') as f:
            for line in f:
                line = line.rstrip()
                if ':' in line and not line.strip().startswith(('-', '!!', '#')):
                    key, val = line.split(':', 1)
                    key = key.strip()
                    val = val.strip()
                    if val and not val.startswith('!!'):
                        try:
                            if val.lower() in ['true', 'false']:
                                config[key] = val.lower() == 'true'
                            elif val == 'null' or val == 'None':
                                config[key] = None
                            else:
                                try:
                                    config[key] = int(val)
                                except ValueError:
                                    try:
                                        config[key] = float(val)
                                    except ValueError:
                                        config[key] = val
                        except:
                            config[key] = val
    except Exception as e:
        print(f"Warning: Could not parse hparams.yaml: {e}")
    return config


def main():
    print("=" * 80)
    print("Logging version_17 (best model: val_Metric=0.605, epoch=112) to WandB")
    print("=" * 80)

    # Check files
    if not VAL_METRICS_FILE.exists():
        print(f"Error: Validation metrics file not found at {VAL_METRICS_FILE}")
        return

    # Load validation metrics
    df_val = pd.read_csv(VAL_METRICS_FILE, index_col=0)
    print(f"\nFound {len(df_val)} epochs of validation data")

    # Compute val_Metric for every epoch
    df_val['val_Metric'] = df_val.apply(compute_val_metric, axis=1)
    best_epoch = int(df_val['val_Metric'].idxmax())
    best_val_metric = df_val['val_Metric'].max()
    print(f"Computed val_Metric — best_epoch: {best_epoch}, best_val_Metric: {best_val_metric:.6f}")

    # Load test metrics
    test_metrics = {}
    if TEST_METRICS_FILE.exists():
        df_test = pd.read_csv(TEST_METRICS_FILE, index_col=0)
        test_metrics = df_test.iloc[0].to_dict()
        print(f"Loaded test metrics from epoch {df_test.index[0]}")

    # Load hyperparameters
    config = load_hparams(HPARAMS_FILE) if HPARAMS_FILE.exists() else {}
    config.update({
        "tasks": TASKS,
        "task_types": TASK_TYPES,
        "task_weights": TASK_WEIGHTS,
        "loss_aggregation": "fixed_weight_sum",
        "model_path": str(MODEL_DIR),
        "best_checkpoint": "best-epoch=112-val_Metric=0.605.ckpt",
    })

    # Initialize wandb
    print("\nInitializing WandB...")
    wandb.init(
        project="MOFSNN_D",
        name="version_17_best_model_0.605",
        group="TSD_SSD_WS24_att_cgcnn",
        tags=["att_cgcnn", "best_model", "version_17"],
        config=config,
        notes="Best model from hyperparameter optimization: epoch=112, val_Metric=0.605. "
              "val_Metric = Σ(per_task_metric × task_weight) using fixed_weight_sum."
    )

    # Log validation metrics for each epoch (including computed val_Metric)
    print("\nLogging validation metrics...")
    for epoch, row in df_val.iterrows():
        metrics_to_log = {"epoch": int(epoch)}

        # Log all individual task metrics
        for col in df_val.columns:
            metrics_to_log[col] = row[col]

        wandb.log(metrics_to_log, step=int(epoch))

        if int(epoch) % 20 == 0 or int(epoch) == best_epoch:
            print(f"  epoch {int(epoch):>4d}  val_Metric={row['val_Metric']:.5f}")

    # Log test metrics as summary
    if test_metrics:
        print("\nLogging test metrics to summary...")
        for key, value in test_metrics.items():
            wandb.run.summary[key] = value
            print(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")

    # Compute test_Metric the same way
    if test_metrics:
        test_metric = 0.0
        for task, task_type, weight in zip(TASKS, TASK_TYPES, TASK_WEIGHTS):
            if task_type == 'regression':
                k = f"{task}/test_R2Score"
            else:
                k = f"{task}/test_BalancedAccuracy"
            if k in test_metrics:
                test_metric += test_metrics[k] * weight
        wandb.run.summary["test_Metric"] = test_metric
        print(f"\n  test_Metric (weighted): {test_metric:.6f}")

    # Summary statistics
    wandb.run.summary.update({
        "best_val_Metric": best_val_metric,
        "best_epoch": best_epoch,
        "total_epochs": len(df_val),
    })

    print("\n" + "=" * 80)
    print("Successfully logged version_17 to WandB!")
    print(f"  Project : MOFSNN_D")
    print(f"  Run     : version_17_best_model_0.605")
    print(f"  URL     : {wandb.run.url}")
    print(f"  best_val_Metric : {best_val_metric:.6f}  (epoch {best_epoch})")
    print("=" * 80)

    wandb.finish()


if __name__ == "__main__":
    main()
