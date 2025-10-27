"""
COPY THIS CODE TO YOUR NOTEBOOKS
This fixes the permission error and configures the path correctly.

INSTRUCTIONS:
1. Open Notebook 1 or Notebook 2 in Jupyter
2. Find the cell that sets DATA_ROOT (usually near the top)
3. Replace that cell with this code
"""

# ============================================
# CONFIGURATION - USE THIS IN NOTEBOOKS 1 & 2
# ============================================

from pathlib import Path
import os

# Set the data root path
DATA_ROOT = Path(r"d:\Users\Dell\Downloads\5th sem proj\demo_dataset")

# Verify the path exists
if not DATA_ROOT.exists():
    raise FileNotFoundError(f"Dataset not found at: {DATA_ROOT}")

# Define class names
CLASS_NAMES = ["Normal", "Asthama", "Pneumonia"]

# Verify all class folders exist
for class_name in CLASS_NAMES:
    class_path = DATA_ROOT / class_name
    if not class_path.exists():
        raise FileNotFoundError(f"Class folder not found: {class_path}")
    
    # Check for .wav files
    wav_files = list(class_path.glob("*.wav"))
    print(f"✓ Found {len(wav_files)} files in {class_name}/")

print(f"\n✓ Dataset configured successfully!")
print(f"  Location: {DATA_ROOT}")
print(f"  Classes: {CLASS_NAMES}")

# Additional configuration (adjust as needed)
FS_TARGET = 44100  # Target sampling rate
FRAME_MS = 250     # Frame length in milliseconds
HOP_RATIO = 0.5    # Overlap ratio
BANDPASS = (250, 2000)  # Bandpass filter range
FILTER_ORDER = 10  # Filter order
SILENCE_PERCENTILE = 20  # Silence removal threshold

print(f"\n✓ Processing parameters configured!")
