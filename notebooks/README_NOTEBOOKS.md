# 📓 Notebooks - Complete Guide

## 🎯 Overview

This folder contains 3 Jupyter notebooks that implement the complete machine learning pipeline for lung sound classification:

1. **`1-spectralpaer-preprocessing.ipynb`** - Audio preprocessing
2. **`2-spectralpaper-feature-extraction.ipynb`** - Feature extraction
3. **`3-spectralpaper-classification.ipynb`** - Model training & evaluation

---

## ⚠️ CRITICAL: Before You Start

### You Need:
1. ✅ **Lung sound dataset** (ICBHI or similar)
2. ✅ **All dependencies installed** (`pip install -r requirements.txt`)
3. ✅ **Jupyter Notebook running** (`python -m notebook`)

### The notebooks are NOT plug-and-play!

They require:
- Dataset path configuration
- Audio files in specific folder structure
- Additional code to save trained models

**👉 Read `NOTEBOOK_SETUP_GUIDE.md` for detailed instructions!**

---

## 🚀 Quick Start (5 Minutes)

### Step 1: Get the Dataset

Download the ICBHI Respiratory Sound Database:
- **Kaggle**: https://www.kaggle.com/datasets/vbookshelf/respiratory-sound-database
- **Official**: https://bhichallenge.med.auth.gr/

Organize it like this:
```
your_dataset/
├── Normal/
├── Asthama/
└── Pneumonia/
```

### Step 2: Configure Notebooks

In **Notebook 1** and **Notebook 2**, update this line:
```python
DATA_ROOT = r"d:\path\to\your\dataset"
```

### Step 3: Run Notebooks in Order

```bash
# Launch Jupyter
cd "d:\Users\Dell\Downloads\5th sem proj\notebooks"
python -m notebook
```

Then run:
1. Notebook 1 → Preprocessing
2. Notebook 2 → Feature Extraction (creates `lung_sound_features.xlsx`)
3. Notebook 3 → Classification

### Step 4: Save Models

**IMPORTANT:** After Notebook 3, add this code in a new cell:

```python
import joblib, os
model_dir = r"d:\Users\Dell\Downloads\5th sem proj\trained_models"
os.makedirs(model_dir, exist_ok=True)

# Save all models (adjust variable names as needed)
try: joblib.dump(ld_model, os.path.join(model_dir, "LD_model.pkl"))
except: pass
try: joblib.dump(gnb_model, os.path.join(model_dir, "GNB_model.pkl"))
except: pass
try: joblib.dump(ft_model, os.path.join(model_dir, "FT_model.pkl"))
except: pass
try: joblib.dump(fknn_model, os.path.join(model_dir, "FKNN_model.pkl"))
except: pass
try: joblib.dump(bt_model, os.path.join(model_dir, "BT_model.pkl"))
except: pass
try: joblib.dump(svm_model, os.path.join(model_dir, "SVM_model.pkl"))
except: pass

print("✅ Models saved!")
```

---

## 📚 Available Documentation

| File | Purpose |
|------|---------|
| **`NOTEBOOK_SETUP_GUIDE.md`** | Complete setup instructions |
| **`ADD_TO_NOTEBOOK3_MODEL_SAVING.py`** | Full model-saving code |
| **`../QUICK_FIX_SUMMARY.md`** | Quick troubleshooting guide |
| **`../AFTER_NOTEBOOKS_GUIDE.md`** | What to do after notebooks |

---

## 🔍 What Each Notebook Does

### Notebook 1: Preprocessing
- Loads raw .wav files
- Normalizes amplitude
- Segments into 250ms frames
- Applies bandpass filter (150-2000 Hz)
- Removes silence
- **Output:** Preprocessed signals (in memory)

### Notebook 2: Feature Extraction
- Takes preprocessed signals
- Extracts 9 spectral features per frame:
  - SCN, SCR, SDC, SEN, SFL, SFLUX, SRO, SSL, SSP
- Creates feature dataset
- **Output:** `lung_sound_features.xlsx`

### Notebook 3: Classification
- Loads feature dataset
- Trains multiple ML models
- Evaluates performance
- **Output:** Trained models (if you add save code)

---

## ✅ Success Checklist

After running all notebooks, you should have:

- [ ] `lung_sound_features.xlsx` file created
- [ ] Trained models saved in `trained_models/` folder
- [ ] Performance metrics showing ~96-98% accuracy
- [ ] Confusion matrices visualized
- [ ] Ready to run GUI application

---

## 🐛 Common Issues

### "No such file or directory"
→ Update `DATA_ROOT` path in notebooks

### "Empty DataFrame"
→ Check that .wav files are in the dataset folders

### "ModuleNotFoundError: openpyxl"
→ Run: `pip install openpyxl`

### "Models not found" in GUI
→ Add model-saving code to Notebook 3

### "Out of memory"
→ Process fewer files or reduce frame overlap

---

## 📊 Expected Results

- **Dataset size:** ~15,000-20,000 feature vectors
- **Best models:** Bagged Trees, Fine Tree (~98% accuracy)
- **SVM accuracy:** ~96-97%
- **Training time:** 5-15 minutes (depends on dataset size)

---

## 🎓 Learning Path

1. **Understand the theory** → Read IEEE paper in `../paper/`
2. **Run Notebook 1** → Learn audio preprocessing
3. **Run Notebook 2** → Understand feature engineering
4. **Run Notebook 3** → Compare ML algorithms
5. **Deploy GUI** → See practical application

---

## 💡 Tips

- **Start small:** Test with a few audio files first
- **Check outputs:** Verify each notebook's output before moving to next
- **Save work:** Notebooks auto-save, but manually save important results
- **Experiment:** Try different parameters and see how results change

---

## 🚀 Next Steps

After successfully running all notebooks:

1. **Run the GUI:**
   ```bash
   cd ../gui_app
   python -m streamlit run spectral-lungsound-gui.py
   ```

2. **Test predictions:** Upload new lung sound files

3. **Improve models:** Try hyperparameter tuning

4. **Deploy:** Share with colleagues or deploy to cloud

---

## 📞 Need Help?

1. Check `NOTEBOOK_SETUP_GUIDE.md` for detailed instructions
2. Review `QUICK_FIX_SUMMARY.md` for common problems
3. Verify all dependencies are installed
4. Make sure dataset is properly organized

---

**Good luck with your lung sound classification project!** 🎉
