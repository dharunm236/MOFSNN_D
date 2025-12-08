import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from CGCNN_MT.inference import inference

# List of CIF files to predict
cif_list = [
    "ACAKUM_clean.cif",
]

model_dir = "./model"
saved_dir = "./predictions"

# Run inference
results = inference(
    cif_list=cif_list,
    model_dir=model_dir,
    saved_dir=saved_dir,
    clean=True  # Whether to clean up temporary files
)

# Print the results
print("\n" + "="*50)
print("Inference Results:")
print("="*50)

# Get list of CIF IDs
cif_ids = results.get("cif_ids", [])

# Iterate through each CIF and print its predictions
for i, cif_id in enumerate(cif_ids):
    print(f"\nMOF: {cif_id}")
    print("-" * 30)
    
    # Iterate through all keys in results to find predictions for this CIF
    for key, value in results.items():
        if key == "cif_ids":
            continue
            
        # Handle numpy arrays
        import numpy as np
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                # Scalar value (single prediction)
                val = value.item()
            else:
                # Array of predictions
                val = value[i]
        elif isinstance(value, list):
             val = value[i]
        else:
             # Fallback for other types
             val = value

        if isinstance(val, (float, int, np.floating, np.integer)):
            print(f"{key}: {val:.4f}")
        else:
            print(f"{key}: {val}")
print("\n" + "="*50)