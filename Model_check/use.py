import os
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from CGCNN_MT.inference import inference
from ML.featuring.feature_generation import descriptor_generator

# --- Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
CIF_DIR = SCRIPT_DIR / "cifs"
MODEL_DIR = SCRIPT_DIR / "model"
RESULTS_DIR = SCRIPT_DIR / "results"
SAVED_DIR = SCRIPT_DIR / "predictions"  # temp dir for graph data
FEAT_DIR = SCRIPT_DIR / "features"      # feature generation working dir

# Ensure output directories exist
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
SAVED_DIR.mkdir(exist_ok=True, parents=True)
FEAT_DIR.mkdir(exist_ok=True, parents=True)

# --- Discover all CIF files ---
cif_list = sorted(CIF_DIR.glob("*.cif"))

if len(cif_list) == 0:
    print(f"No CIF files found in {CIF_DIR}")
    sys.exit(1)

print(f"Found {len(cif_list)} CIF file(s) in {CIF_DIR}:")
for cif in cif_list:
    print(f"  - {cif.name}")

# --- Step 1: Compute RAC + Zeo++ extra features for each CIF ---
print("\n" + "=" * 60)
print("Step 1: Computing RAC + Zeo++ features...")
print("=" * 60)

# descriptor_generator creates sub-folders under feature_folders/<name>/
# We must run from the ML/featuring dir since RAC_getter.py is called as a subprocess
# with a relative path, and feature_folders are created relative to CWD
FEATURING_DIR = Path(PROJECT_ROOT) / "ML" / "featuring"
orig_cwd = os.getcwd()
os.chdir(FEATURING_DIR)
if not os.path.exists("feature_folders"):
    os.mkdir("feature_folders")

WIGGLE_ROOM = 1.0
PROB_RADIUS = 1.86  # same as used during training (see notebooks)

for cif_path in cif_list:
    mof_name = cif_path.stem
    merged_csv = f"feature_folders/{mof_name}/merged_descriptors/{mof_name}_descriptors.csv"
    if os.path.exists(merged_csv):
        print(f"  Features already computed for {mof_name}, skipping.")
        continue
    print(f"  Computing features for {mof_name}...")
    descriptor_generator(mof_name, str(cif_path), WIGGLE_ROOM, PROB_RADIUS)

# Collect all per-MOF feature CSVs into a single DataFrame
all_dfs = []
unsuccessful = []
for cif_path in cif_list:
    mof_name = cif_path.stem
    csv_path = f"feature_folders/{mof_name}/merged_descriptors/{mof_name}_descriptors.csv"
    if os.path.exists(csv_path):
        df_feat = pd.read_csv(csv_path)
        all_dfs.append(df_feat)
    else:
        unsuccessful.append(mof_name)
        print(f"  WARNING: Feature generation failed for {mof_name}")

os.chdir(orig_cwd)

if unsuccessful:
    print(f"\nFailed featurizations: {unsuccessful}")

if all_dfs:
    extra_fea_df = pd.concat(all_dfs, ignore_index=True)
    extra_fea_df.drop(columns=["cif_file"], inplace=True, errors="ignore")
    extra_fea_df.rename(columns={"name": "MofName"}, inplace=True)
    extra_fea_df.set_index("MofName", inplace=True)

    # Enforce the exact 190 feature columns in the same order as training data.
    # Different CIFs can produce RAC columns in different orders; pd.concat then
    # creates a superset with NaN, leading to shape mismatches at inference time.
    ref_csv = Path(PROJECT_ROOT) / "CGCNN_MT" / "data" / "TSD" / "RAC_and_zeo_features_with_id_prop.csv"
    ref_cols = pd.read_csv(ref_csv, nrows=0).columns.tolist()
    # Training features start from the "Di" column onwards (after MofName, Label, Partition)
    di_idx = ref_cols.index("Di")
    expected_feat_cols = ref_cols[di_idx:]
    # Keep only the expected columns (in the right order), fill missing with 0
    extra_fea_df = extra_fea_df.reindex(columns=expected_feat_cols, fill_value=0)
    extra_fea_df.fillna(0, inplace=True)

    # Save features CSV for reference
    extra_fea_df.to_csv(RESULTS_DIR / "computed_features.csv")
    print(f"\nFeatures computed for {len(extra_fea_df)} MOFs. Shape: {extra_fea_df.shape}")
else:
    extra_fea_df = None
    print("\nWARNING: No features could be computed. Inference will proceed without extra features.")

# --- Step 2: Run inference ---
print("\n" + "=" * 60)
print("Step 2: Running model inference...")
print("=" * 60)

results = inference(
    cif_list=cif_list,
    model_dir=str(MODEL_DIR),
    saved_dir=str(SAVED_DIR),
    extra_fea_df=extra_fea_df,
    clean=True,
)

# --- Step 3: Build results DataFrame ---
cif_ids = results.get("cif_ids", [])

# Collect prediction and probability columns
data = {"MOF_Name": cif_ids}
for key, value in results.items():
    if key == "cif_ids":
        continue
    # Include prediction, probability, and uncertainty columns
    if key.endswith("_pred") or "_prob" in key or "_uncertainty" in key:
        if isinstance(value, np.ndarray):
            if value.ndim == 1:
                data[key] = value
            elif value.ndim == 2:
                # For multi-class probabilities, store each class as a separate column
                for cls_idx in range(value.shape[1]):
                    data[f"{key}_class{cls_idx}"] = value[:, cls_idx]
            else:
                data[key] = value.squeeze()
        elif isinstance(value, list):
            data[key] = value

df = pd.DataFrame(data)

# --- Save to CSV ---
output_csv = RESULTS_DIR / "inference_results.csv"
df.to_csv(output_csv, index=False, float_format="%.4f")
print(f"\nResults saved to {output_csv}")

# --- Save human-readable text report ---
output_txt = RESULTS_DIR / "inference_results.txt"

# Task metadata for readable interpretation
TASK_INFO = {
    "TSD": {"type": "regression", "unit": "°C", "description": "Thermal Stability Decomposition Temperature"},
    "SSD": {"type": "classification", "classes": {0: "Unstable", 1: "Stable"}, "description": "Solvent Stability (binary)"},
    "WS24_water": {"type": "classification", "classes": {0: "Unstable", 1: "Stable"}, "description": "Water Stability (24h)"},
    "WS24_water4": {"type": "classification_4", "classes": {0: "Class 0", 1: "Class 1", 2: "Class 2", 3: "Class 3"}, "description": "Water Stability (4-class)"},
    "WS24_acid": {"type": "classification", "classes": {0: "Unstable", 1: "Stable"}, "description": "Acid Stability (24h)"},
    "WS24_base": {"type": "classification", "classes": {0: "Unstable", 1: "Stable"}, "description": "Base Stability (24h)"},
    "WS24_boiling": {"type": "classification", "classes": {0: "Unstable", 1: "Stable"}, "description": "Boiling Water Stability (24h)"},
}

with open(output_txt, "w") as f:
    f.write("=" * 70 + "\n")
    f.write("  MOF Stability Neural Network — Inference Results\n")
    f.write(f"  Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"  Model: {MODEL_DIR}\n")
    f.write(f"  CIF directory: {CIF_DIR}\n")
    f.write(f"  Number of MOFs: {len(cif_ids)}\n")
    f.write("=" * 70 + "\n\n")

    for i, cif_id in enumerate(cif_ids):
        f.write("-" * 70 + "\n")
        f.write(f"  MOF: {cif_id}\n")
        f.write("-" * 70 + "\n\n")

        for task, info in TASK_INFO.items():
            pred_key = f"{task}_pred"
            if pred_key not in df.columns:
                continue

            pred_val = df[pred_key].iloc[i]
            f.write(f"  {info['description']} ({task}):\n")

            if info["type"] == "regression":
                f.write(f"    Predicted value : {pred_val:.2f} {info.get('unit', '')}\n")
            else:
                pred_label = info["classes"].get(int(pred_val), f"Class {int(pred_val)}")
                f.write(f"    Prediction      : {pred_label} (class {int(pred_val)})\n")

                # Write class probabilities
                prob_cols = [c for c in df.columns if c.startswith(f"{task}_prob")]
                if prob_cols:
                    f.write(f"    Probabilities   : ")
                    prob_parts = []
                    for pc in prob_cols:
                        cls_idx = int(pc.split("class")[-1])
                        cls_label = info["classes"].get(cls_idx, f"Class {cls_idx}")
                        prob_parts.append(f"{cls_label}={df[pc].iloc[i]:.4f}")
                    f.write(", ".join(prob_parts) + "\n")
            f.write("\n")
        f.write("\n")

    # Summary table at the end
    f.write("=" * 70 + "\n")
    f.write("  Summary Table\n")
    f.write("=" * 70 + "\n\n")

    # Compact summary: one row per MOF with key predictions
    header = f"{'MOF':<15} {'TSD(°C)':>10} {'SSD':>6} {'Water':>6} {'Water4':>7} {'Acid':>6} {'Base':>6} {'Boiling':>8}\n"
    f.write(header)
    f.write("-" * 70 + "\n")
    for i, cif_id in enumerate(cif_ids):
        tsd = df["TSD_pred"].iloc[i] if "TSD_pred" in df.columns else float("nan")
        ssd = int(df["SSD_pred"].iloc[i]) if "SSD_pred" in df.columns else "-"
        water = int(df["WS24_water_pred"].iloc[i]) if "WS24_water_pred" in df.columns else "-"
        water4 = int(df["WS24_water4_pred"].iloc[i]) if "WS24_water4_pred" in df.columns else "-"
        acid = int(df["WS24_acid_pred"].iloc[i]) if "WS24_acid_pred" in df.columns else "-"
        base = int(df["WS24_base_pred"].iloc[i]) if "WS24_base_pred" in df.columns else "-"
        boiling = int(df["WS24_boiling_pred"].iloc[i]) if "WS24_boiling_pred" in df.columns else "-"
        f.write(f"{cif_id:<15} {tsd:>10.2f} {ssd:>6} {water:>6} {water4:>7} {acid:>6} {base:>6} {boiling:>8}\n")

    f.write("-" * 70 + "\n")
    f.write("\nLegend:\n")
    f.write("  TSD     = Thermal decomposition temperature (°C)\n")
    f.write("  SSD     = Solvent stability: 0=Unstable, 1=Stable\n")
    f.write("  Water   = Water stability (24h): 0=Unstable, 1=Stable\n")
    f.write("  Water4  = Water stability (4-class): 0-3\n")
    f.write("  Acid    = Acid stability (24h): 0=Unstable, 1=Stable\n")
    f.write("  Base    = Base stability (24h): 0=Unstable, 1=Stable\n")
    f.write("  Boiling = Boiling water stability (24h): 0=Unstable, 1=Stable\n")

print(f"Text report saved to {output_txt}")

# --- Print summary ---
print("\n" + "=" * 60)
print("Inference Results Summary")
print("=" * 60)
print(df.to_string(index=False))
print("=" * 60)