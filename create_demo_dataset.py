"""
Create Synthetic Lung Sound Dataset for Demo
This generates fake lung sound data so you can test the system without downloading the full dataset.
"""

import numpy as np
import os
from scipy.io import wavfile
from tqdm import tqdm

print("=" * 60)
print("CREATING DEMO LUNG SOUND DATASET")
print("=" * 60)

# Configuration
OUTPUT_DIR = r"d:\Users\Dell\Downloads\5th sem proj\demo_dataset"
SAMPLE_RATE = 44100  # Hz
DURATION = 5  # seconds per file
NUM_FILES_PER_CLASS = 10  # Number of files per class

# Create output directories
classes = ["Normal", "Asthama", "Pneumonia"]
for class_name in classes:
    class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(class_dir, exist_ok=True)

print(f"\nOutput directory: {OUTPUT_DIR}")
print(f"Sample rate: {SAMPLE_RATE} Hz")
print(f"Duration: {DURATION} seconds per file")
print(f"Files per class: {NUM_FILES_PER_CLASS}")
print()

def generate_normal_sound(duration, sr):
    """Generate synthetic normal lung sound (regular breathing)"""
    t = np.linspace(0, duration, int(sr * duration))
    
    # Base breathing frequency (12-20 breaths per minute)
    breath_freq = 0.3  # Hz (18 breaths/min)
    
    # Normal breathing pattern - smooth sine wave
    signal = 0.3 * np.sin(2 * np.pi * breath_freq * t)
    
    # Add some harmonics for realism
    signal += 0.1 * np.sin(2 * np.pi * 2 * breath_freq * t)
    signal += 0.05 * np.sin(2 * np.pi * 3 * breath_freq * t)
    
    # Add slight noise
    noise = 0.02 * np.random.randn(len(t))
    signal += noise
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    return signal

def generate_asthma_sound(duration, sr):
    """Generate synthetic asthma lung sound (wheezing)"""
    t = np.linspace(0, duration, int(sr * duration))
    
    # Faster, irregular breathing
    breath_freq = 0.4  # Hz (24 breaths/min)
    
    # Wheezing - high-frequency component
    wheeze_freq = 800  # Hz
    signal = 0.2 * np.sin(2 * np.pi * breath_freq * t)
    signal += 0.3 * np.sin(2 * np.pi * wheeze_freq * t) * (0.5 + 0.5 * np.sin(2 * np.pi * breath_freq * t))
    
    # Add harmonics
    signal += 0.15 * np.sin(2 * np.pi * 2 * breath_freq * t)
    
    # More noise (turbulent airflow)
    noise = 0.05 * np.random.randn(len(t))
    signal += noise
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    return signal

def generate_pneumonia_sound(duration, sr):
    """Generate synthetic pneumonia lung sound (crackles)"""
    t = np.linspace(0, duration, int(sr * duration))
    
    # Shallow, rapid breathing
    breath_freq = 0.5  # Hz (30 breaths/min)
    
    # Base breathing
    signal = 0.25 * np.sin(2 * np.pi * breath_freq * t)
    
    # Add crackles (random impulses)
    num_crackles = 20
    crackle_positions = np.random.randint(0, len(t), num_crackles)
    for pos in crackle_positions:
        if pos < len(t) - 100:
            # Short impulse
            crackle = np.exp(-np.arange(100) / 10) * np.random.randn(100)
            signal[pos:pos+100] += 0.4 * crackle
    
    # Add more noise (fluid in lungs)
    noise = 0.08 * np.random.randn(len(t))
    signal += noise
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    return signal

# Generate files for each class
generators = {
    "Normal": generate_normal_sound,
    "Asthama": generate_asthma_sound,
    "Pneumonia": generate_pneumonia_sound
}

print("Generating synthetic lung sounds...\n")

for class_name in classes:
    print(f"Creating {class_name} sounds...")
    class_dir = os.path.join(OUTPUT_DIR, class_name)
    generator = generators[class_name]
    
    for i in tqdm(range(NUM_FILES_PER_CLASS), desc=f"  {class_name}"):
        # Generate signal
        signal = generator(DURATION, SAMPLE_RATE)
        
        # Convert to int16 for WAV format
        signal_int16 = np.int16(signal * 32767)
        
        # Save as WAV file
        filename = f"{class_name.lower()}_{i+1:03d}.wav"
        filepath = os.path.join(class_dir, filename)
        wavfile.write(filepath, SAMPLE_RATE, signal_int16)
    
    print(f"  Created {NUM_FILES_PER_CLASS} files in {class_dir}\n")

print("=" * 60)
print("DEMO DATASET CREATED SUCCESSFULLY!")
print("=" * 60)
print(f"\nLocation: {OUTPUT_DIR}")
print(f"Total files: {NUM_FILES_PER_CLASS * len(classes)}")
print("\nNext steps:")
print("1. Update notebooks with this path:")
print(f"   DATA_ROOT = r\"{OUTPUT_DIR}\"")
print("2. Run the notebooks in order (1 -> 2 -> 3)")
print("3. The system will work with this demo data!")
print("\nNote: This is synthetic data for demo purposes.")
print("   For real research, use the actual ICBHI dataset.")
print("=" * 60)
