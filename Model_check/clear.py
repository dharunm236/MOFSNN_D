"""
Clears all generated output folders:
  - results/
  - predictions/
  - features/
"""

import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

FOLDERS = [
    SCRIPT_DIR / "results",
    SCRIPT_DIR / "predictions",
    SCRIPT_DIR / "features",
]

for folder in FOLDERS:
    if folder.exists():
        shutil.rmtree(folder)
        folder.mkdir()
        print(f"Cleared: {folder}")
    else:
        print(f"Skipped (not found): {folder}")

print("\nDone.")
