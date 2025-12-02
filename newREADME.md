# MOFSNN_D Setup Guide for Ubuntu GCP

This guide provides step-by-step instructions to set up and run the MOFSNN_D project on an Ubuntu GCP instance.

## Prerequisites

- Ubuntu 20.04/22.04 LTS on GCP
- At least 32GB RAM recommended
- GPU with CUDA support (for training)
- At least 100GB disk space

---

## 1. System Dependencies

```bash
# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install essential build tools
sudo apt-get install -y build-essential wget curl git unzip
sudo apt-get install -y libxrender1 libxext6 libsm6 libgl1-mesa-glx
sudo apt-get install -y cmake g++ gfortran

# Install additional dependencies for pymatgen and ASE
sudo apt-get install -y libopenblas-dev liblapack-dev
```

---

## 2. Install Miniconda

```bash
# Download Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

# Install Miniconda
bash ~/miniconda.sh -b -p $HOME/miniconda3

# Initialize conda
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init bash

# Reload shell
source ~/.bashrc
```

---

## 3. Create Conda Environment

```bash
# Create new environment with Python 3.9
conda create -n mofsnn python=3.9 -y

# Activate environment
conda activate mofsnn
```

---

## 4. Install Python Dependencies

```bash
# Install PyTorch with CUDA support (adjust CUDA version as needed)
# For CUDA 11.8:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# OR for CUDA 12.1:
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install PyTorch Lightning
pip install pytorch-lightning==2.0.9

# Install core scientific packages
conda install -c conda-forge numpy pandas scipy matplotlib seaborn -y
pip install scikit-learn

# Install materials science packages
pip install pymatgen==2024.2.8
pip install ase==3.22.1

# Install Jupyter
pip install jupyter notebook ipywidgets

# Install additional dependencies
pip install tqdm
pip install optuna  # For hyperparameter optimization

# Install torch-geometric (required for graph neural networks)
pip install torch-geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Install molSimplify for RAC features (if needed)
pip install molSimplify
```

---

## 5. Install Zeo++ (Critical for Feature Generation)

Zeo++ is required for generating zeolitic features from CIF files.

```bash
# Navigate to home directory
cd ~

# Download Zeo++ source
wget http://www.zeoplusplus.org/zeo++-0.3.tar.gz

# Extract
tar -xzf zeo++-0.3.tar.gz
cd zeo++-0.3

# Compile Zeo++
make

# Verify installation
./network -h

# Add Zeo++ to PATH permanently
echo 'export PATH="$HOME/zeo++-0.3:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify it's accessible
which network
```

### Alternative: Install via Conda (if above fails)

```bash
conda install -c conda-forge zeopp-lsmo -y
```

---

## 6. Navigate to Project Directory

```bash
# Assuming you cloned the repo to home directory
cd ~/MOFSNN_D

# Or wherever you cloned it
# cd /path/to/MOFSNN_D
```

---

## 7. Download Required Datasets

### 7.1 Download CoREMOF2019 Database

```bash
# Create raw_data directory
mkdir -p raw_data

# Download CoREMOF 2019 dataset
cd raw_data
wget "https://mof.tech.northwestern.edu/Datasets/CoREMOF%202019-mofdb-version:dc8a0295db.zip" -O CoREMOF_2019.zip

# Unzip (may take a while)
unzip -o CoREMOF_2019.zip -d CoREMOF2019

cd ..
```

### 7.2 Verify SciData Files Exist

The TSD and SSD data should already be in the repository:

```bash
# Check if the files exist
ls -la raw_data/SciData/separate_files/solvent_removal_stability/full_SSD_data.csv
ls -la raw_data/SciData/separate_files/thermal_stability/full_TSD_data.csv

# If not present, you need to obtain these from the original source
```

---

## 8. Update Paths in Code (if necessary)

### 8.1 Check and Update ML/featuring Path

The feature generation script path needs to be correct:

```bash
# Verify the feature_generation.py exists
ls -la ML/featuring/feature_generation.py

# If the path is different, note it for later
```

### 8.2 Update feature_generation.py Zeo++ Path (if needed)

```bash
# Open the feature generation script
nano ML/featuring/feature_generation.py

# Look for any hardcoded zeo++ paths and update them
# The 'network' command should be in your PATH now
```

---

## 9. Create Required Directories

```bash
# Create all necessary directories
mkdir -p CGCNN_MT/data/SSD/cifs
mkdir -p CGCNN_MT/data/SSD/clean_cifs
mkdir -p CGCNN_MT/data/SSD/features

mkdir -p CGCNN_MT/data/TSD/cifs
mkdir -p CGCNN_MT/data/TSD/clean_cifs
mkdir -p CGCNN_MT/data/TSD/features

mkdir -p CGCNN_MT/data/WS24/cifs
mkdir -p CGCNN_MT/data/WS24/clean_cifs
mkdir -p CGCNN_MT/data/WS24/features

mkdir -p CGCNN_MT/logs
mkdir -p CGCNN_MT/evaluation
```

---

## 10. Running the Notebooks

### Option A: Run via Jupyter Notebook (Recommended for debugging)

```bash
# Start Jupyter with no browser (for remote access)
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0

# If you need to tunnel from local machine:
# On your LOCAL machine, run:
# ssh -L 8888:localhost:8888 your-username@your-gcp-external-ip
```

### Option B: Convert Notebooks to Python Scripts

```bash
# Install nbconvert
pip install nbconvert

# Convert notebooks to Python scripts
jupyter nbconvert --to script 01_process_TSDandSSD.ipynb
jupyter nbconvert --to script 02_process_WS24.ipynb  
jupyter nbconvert --to script 07_CGCNN_train.ipynb

# Run scripts sequentially
python 01_process_TSDandSSD.py
python 02_process_WS24.py
python 07_CGCNN_train.py
```

---

## 11. Step-by-Step Notebook Execution

### 11.1 Running 01_process_TSDandSSD.ipynb

This notebook processes the Thermal Stability Dataset (TSD) and Solvent Stability Dataset (SSD).

**Before running, verify:**

```bash
# Check source files exist
ls raw_data/SciData/separate_files/solvent_removal_stability/full_SSD_data.csv
ls raw_data/SciData/separate_files/thermal_stability/full_TSD_data.csv
ls raw_data/CoREMOF2019/*.cif | head -5  # Should show CIF files
```

**Key cells to modify (if needed):**

```python
# Cell with paths - update if your structure is different
ssd_csv = "./raw_data/SciData/separate_files/solvent_removal_stability/full_SSD_data.csv"
tsd_csv = "./raw_data/SciData/separate_files/thermal_stability/full_TSD_data.csv"
saved_dir = Path("./CGCNN_MT/data/")
src_cif_dir = Path("./raw_data/CoREMOF2019")
```

**Expected outputs:**
- `CGCNN_MT/data/SSD/id_prop_feat.csv`
- `CGCNN_MT/data/TSD/id_prop_feat.csv`
- `CGCNN_MT/data/SSD/RAC_and_zeo_features_with_id_prop.csv`
- `CGCNN_MT/data/TSD/RAC_and_zeo_features_with_id_prop.csv`
- `.graphdata` files in clean_cifs directories

---

### 11.2 Running 02_process_WS24.ipynb

This notebook processes the Water Stability 2024 dataset.

**Before running, verify:**

```bash
# Check WS24 source files exist (adjust path based on your data)
ls raw_data/WS24/  # or wherever your WS24 data is located
```

**Expected outputs:**
- `CGCNN_MT/data/WS24/id_prop_feat.csv`
- `CGCNN_MT/data/WS24/RAC_and_zeo_features_with_id_prop.csv`

---

### 11.3 Running 07_CGCNN_train.ipynb

This notebook trains the CGCNN model.

**Before running, verify all data is processed:**

```bash
# Verify all required data files exist
ls CGCNN_MT/data/SSD/RAC_and_zeo_features_with_id_prop.csv
ls CGCNN_MT/data/TSD/RAC_and_zeo_features_with_id_prop.csv
ls CGCNN_MT/data/WS24/RAC_and_zeo_features_with_id_prop.csv

# Verify graph data exists
ls CGCNN_MT/data/SSD/clean_cifs/*.graphdata | head -5
ls CGCNN_MT/data/TSD/clean_cifs/*.graphdata | head -5
```

**For GPU training, verify CUDA:**

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

---

## 12. Troubleshooting Common Issues

### Issue 1: Zeo++ not found

```bash
# Check if network command is available
which network

# If not, add to PATH
export PATH="$HOME/zeo++-0.3:$PATH"

# Or use full path in feature_generation.py
```

### Issue 2: CUDA out of memory

```python
# In training notebook, reduce batch size
batch_size = 16  # or 8
```

### Issue 3: FileNotFoundError for CIF files

```bash
# Verify CIF files were copied correctly
ls CGCNN_MT/data/SSD/cifs/ | wc -l
ls CGCNN_MT/data/TSD/cifs/ | wc -l

# If empty, re-run the CIF copying cells in notebook
```

### Issue 4: ModuleNotFoundError

```bash
# Make sure you're in the correct conda environment
conda activate mofsnn

# Install missing package
pip install <package_name>
```

### Issue 5: Permission denied for feature_generation.py

```bash
chmod +x ML/featuring/feature_generation.py
```

### Issue 6: pymatgen CIF reading warnings

These are usually harmless warnings. They're already suppressed in the notebooks:

```python
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen.io.cif")
```

---

## 13. Expected Runtime

| Notebook | Approximate Time |
|----------|------------------|
| 01_process_TSDandSSD.ipynb | 3-5 hours (feature generation is slow) |
| 02_process_WS24.ipynb | 1-2 hours |
| 07_CGCNN_train.ipynb | 2-8 hours (depends on GPU and epochs) |

---

## 14. Verification Checklist

Before running training, verify all files exist:

```bash
#!/bin/bash
# Save this as verify_setup.sh and run it

echo "=== Checking Environment ==="
conda activate mofsnn
python --version
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import pytorch_lightning; print(f'PyTorch Lightning: {pytorch_lightning.__version__}')"
python -c "import pymatgen; print(f'Pymatgen: {pymatgen.__version__}')"

echo ""
echo "=== Checking Zeo++ ==="
which network || echo "WARNING: Zeo++ not in PATH"

echo ""
echo "=== Checking Data Files ==="
for f in \
    "raw_data/SciData/separate_files/solvent_removal_stability/full_SSD_data.csv" \
    "raw_data/SciData/separate_files/thermal_stability/full_TSD_data.csv" \
    "CGCNN_MT/data/SSD/id_prop_feat.csv" \
    "CGCNN_MT/data/TSD/id_prop_feat.csv"
do
    if [ -f "$f" ]; then
        echo "✓ $f exists"
    else
        echo "✗ $f MISSING"
    fi
done

echo ""
echo "=== Checking CIF Directories ==="
echo "CoREMOF2019 CIFs: $(ls raw_data/CoREMOF2019/*.cif 2>/dev/null | wc -l)"
echo "SSD CIFs: $(ls CGCNN_MT/data/SSD/cifs/*.cif 2>/dev/null | wc -l)"
echo "TSD CIFs: $(ls CGCNN_MT/data/TSD/cifs/*.cif 2>/dev/null | wc -l)"

echo ""
echo "=== Setup verification complete ==="
```

---

## 15. Quick Start Commands Summary

```bash
# 1. Activate environment
conda activate mofsnn

# 2. Navigate to project
cd ~/MOFSNN_D

# 3. Start Jupyter
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0

# 4. Run notebooks in order:
#    - 01_process_TSDandSSD.ipynb
#    - 02_process_WS24.ipynb
#    - 07_CGCNN_train.ipynb
```

---

## Notes

- The feature generation step (Zeo++ and RAC calculation) is CPU-intensive and can take several hours.
- Consider using `screen` or `tmux` for long-running processes to prevent disconnection issues.
- Monitor GPU memory usage with `nvidia-smi` during training.

```bash
# Install screen for persistent sessions
sudo apt-get install screen -y

# Start a new screen session
screen -S moftraining

# Detach with Ctrl+A, then D
# Reattach with: screen -r moftraining
```