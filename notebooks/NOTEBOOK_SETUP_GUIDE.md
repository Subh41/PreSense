# 📓 Notebook Setup & Execution Guide

## ⚠️ IMPORTANT: Read This Before Running Notebooks

The notebooks in this folder are designed to work with the **ICBHI Respiratory Sound Database** or similar lung sound datasets. You need to set up your data properly before running them.

---

## 🗂️ Step 1: Prepare Your Dataset

### Required Dataset Structure:

```
your_dataset_folder/
├── Normal/          # Normal lung sounds (.wav files)
├── Asthama/         # Asthma lung sounds (.wav files)
└── Pneumonia/       # Pneumonia lung sounds (.wav files)
```

### Where to Get the Dataset:

1. **ICBHI Respiratory Sound Database** (Recommended):
   - Download from: https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database
   - Or: https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge

2. **Alternative Sources**:
   - Search for "lung sound dataset" on Kaggle
   - Use your own collected lung sound recordings

---

## 🔧 Step 2: Configure the Notebooks

### For Notebook 1 & 2 (Preprocessing & Feature Extraction):

1. Open the notebook in Jupyter
2. Find the cell with `DATA_ROOT` variable (usually in the first code cell)
3. Update it to point to your dataset folder:

```python
# CHANGE THIS PATH TO YOUR DATASET LOCATION
DATA_ROOT = r"d:\path\to\your\lung_sound_dataset"

# Example:
# DATA_ROOT = r"d:\Users\Dell\Downloads\respiratory-sound-database"
```

4. Make sure your dataset has these three folders:
   - `Normal/` (or `normal/`)
   - `Asthama/` (or `asthma/`)
   - `Pneumonia/` (or `pneumonia/`)

---

## 📝 Step 3: Run the Notebooks in Order

### **Notebook 1: Preprocessing**

**Purpose:** Load, normalize, filter, and segment lung sounds

**What to do:**
1. Update `DATA_ROOT` path
2. Run all cells sequentially
3. Check the visualizations (time/frequency domain plots)

**Expected Output:**
- Preprocessed signals stored in memory
- Visualization plots
- No files saved (data passed to next notebook)

**Common Issues:**
- ❌ "No such file or directory" → Check `DATA_ROOT` path
- ❌ "Empty folder" → Make sure .wav files are in the class folders
- ❌ "Memory error" → Reduce the number of files or process in batches

---

### **Notebook 2: Feature Extraction**

**Purpose:** Extract 9 spectral features from preprocessed signals

**What to do:**
1. Update `DATA_ROOT` path (same as Notebook 1)
2. Run all cells sequentially
3. Features will be saved to `lung_sound_features.xlsx`

**Expected Output:**
- Excel file: `lung_sound_features.xlsx` with columns:
  - SCN, SCR, SDC, SEN, SFL, SFLUX, SRO, SSL, SSP, Class, File
- Scatter plots showing feature separability
- Correlation heatmaps

**Common Issues:**
- ❌ "ModuleNotFoundError: openpyxl" → Run: `pip install openpyxl`
- ❌ "No data to extract" → Make sure Notebook 1 preprocessing worked
- ❌ File not saving → Check write permissions in the notebook directory

---

### **Notebook 3: Classification**

**Purpose:** Train ML models and evaluate performance

**What to do:**
1. Make sure `lung_sound_features.xlsx` exists (from Notebook 2)
2. Run all cells sequentially
3. **IMPORTANT:** Add model-saving code at the end (see below)

**Expected Output:**
- Trained models saved as `.pkl` files
- Confusion matrices for each model
- Performance comparison charts
- Statistical test results

**⭐ CRITICAL: Add This Code at the End of Notebook 3:**

```python
# ============================================
# SAVE ALL TRAINED MODELS
# ============================================
import joblib
import os

# Create directory for trained models
model_dir = r"d:\Users\Dell\Downloads\5th sem proj\trained_models"
os.makedirs(model_dir, exist_ok=True)

# Save all models (adjust variable names based on your notebook)
# The variable names depend on how you named them in the notebook

# Example - adjust these variable names to match your notebook:
models_to_save = {
    "LD_model.pkl": ld_model,           # Linear Discriminant
    "GNB_model.pkl": gnb_model,         # Gaussian Naive Bayes
    "KNB_model.pkl": knb_model,         # Kernel Naive Bayes
    "FT_model.pkl": ft_model,           # Fine Tree
    "FKNN_model.pkl": fknn_model,       # Fine KNN
    "BT_model.pkl": bt_model,           # Bagged Trees
    "SKNN_model.pkl": sknn_model,       # Subspace KNN
    "SVM_model.pkl": svm_model,         # SVM (best model)
}

# Save each model
for filename, model in models_to_save.items():
    filepath = os.path.join(model_dir, filename)
    joblib.dump(model, filepath)
    print(f"✅ Saved: {filename}")

print(f"\n🎉 All models saved to: {model_dir}")
```

**Common Issues:**
- ❌ "File not found: lung_sound_features.xlsx" → Run Notebook 2 first
- ❌ "NameError: model not defined" → Check variable names in the save code
- ❌ Models not loading in GUI → Make sure file names end with `_model.pkl`

---

## 🎯 Step 4: After Running All Notebooks

Once you've successfully run all three notebooks:

1. ✅ Check that `lung_sound_features.xlsx` exists
2. ✅ Check that `trained_models/` folder has `.pkl` files
3. ✅ Review the performance metrics (accuracy, F1-score)
4. ✅ Identify the best performing model

**Now you can:**
- Run the GUI application (see main README.md)
- Use the trained models for predictions
- Deploy the system

---

## 🐛 Troubleshooting

### Problem: "No audio files found"
**Solution:** 
- Check that .wav files are in the correct folders
- Verify folder names match exactly: `Normal`, `Asthama`, `Pneumonia`
- Check file permissions

### Problem: "Out of memory"
**Solution:**
- Process fewer files at a time
- Reduce frame overlap in preprocessing
- Close other applications

### Problem: "Features file not found"
**Solution:**
- Make sure Notebook 2 completed successfully
- Check that `lung_sound_features.xlsx` was created
- Look in the notebooks directory

### Problem: "Model variable not found when saving"
**Solution:**
- Check the variable names in your Notebook 3
- Update the `models_to_save` dictionary with correct names
- Make sure all models were trained successfully

### Problem: "GUI can't find models"
**Solution:**
- Verify models are saved in `trained_models/` folder
- Check file names end with `_model.pkl`
- Verify the path in `gui_app/spectral-lungsound-gui.py`

---

## 📊 Expected Results

After running all notebooks successfully:

- **Preprocessing:** Clean, filtered signals ready for analysis
- **Feature Extraction:** ~15,000-20,000 feature vectors (depends on dataset size)
- **Classification:** 
  - Best models: Bagged Trees, Fine Tree (~98% accuracy)
  - SVM: ~96-97% accuracy
  - Other models: 60-85% accuracy

---

## 📚 Additional Resources

- **IEEE Paper:** See `paper/` folder for full methodology
- **Video Tutorials:** Check main README.md for YouTube links
- **Dataset Info:** ICBHI database documentation

---

## ✅ Quick Checklist

Before running notebooks:
- [ ] Dataset downloaded and organized
- [ ] `DATA_ROOT` path updated in notebooks
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Jupyter Notebook running (`python -m notebook`)

After running notebooks:
- [ ] `lung_sound_features.xlsx` created
- [ ] Trained models saved in `trained_models/` folder
- [ ] Performance metrics reviewed
- [ ] Ready to run GUI application

---

**Need Help?** Check the main `AFTER_NOTEBOOKS_GUIDE.md` for post-processing steps!
