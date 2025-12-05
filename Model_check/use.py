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