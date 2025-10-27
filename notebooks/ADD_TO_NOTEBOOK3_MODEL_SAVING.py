"""
IMPORTANT: Add this code at the END of Notebook 3 (Classification)
This will save all your trained models so the GUI can use them.

Copy and paste this entire cell into a new cell at the end of Notebook 3.
"""

# ============================================
# SAVE ALL TRAINED MODELS FOR GUI
# ============================================
import joblib
import os

print("=" * 60)
print("SAVING TRAINED MODELS")
print("=" * 60)

# Create directory for trained models
model_dir = r"d:\Users\Dell\Downloads\5th sem proj\trained_models"
os.makedirs(model_dir, exist_ok=True)
print(f"📁 Model directory: {model_dir}\n")

# ============================================
# IMPORTANT: Update these variable names to match your notebook!
# Look at your Notebook 3 code and find the variable names you used
# when training each model.
# ============================================

# Dictionary of models to save
# Format: "filename": variable_name
models_to_save = {}

# Try to save each model if it exists
# Linear Discriminant Analysis
try:
    models_to_save["LD_model.pkl"] = ld_model
    print("✅ Found: Linear Discriminant (ld_model)")
except NameError:
    print("⚠️  Not found: ld_model (Linear Discriminant)")

# Gaussian Naive Bayes
try:
    models_to_save["GNB_model.pkl"] = gnb_model
    print("✅ Found: Gaussian Naive Bayes (gnb_model)")
except NameError:
    print("⚠️  Not found: gnb_model (Gaussian Naive Bayes)")

# Kernel Naive Bayes
try:
    models_to_save["KNB_model.pkl"] = knb_model
    print("✅ Found: Kernel Naive Bayes (knb_model)")
except NameError:
    print("⚠️  Not found: knb_model (Kernel Naive Bayes)")

# Fine Tree
try:
    models_to_save["FT_model.pkl"] = ft_model
    print("✅ Found: Fine Tree (ft_model)")
except NameError:
    print("⚠️  Not found: ft_model (Fine Tree)")

# Fine K-Nearest Neighbors
try:
    models_to_save["FKNN_model.pkl"] = fknn_model
    print("✅ Found: Fine KNN (fknn_model)")
except NameError:
    print("⚠️  Not found: fknn_model (Fine KNN)")

# Bagged Trees
try:
    models_to_save["BT_model.pkl"] = bt_model
    print("✅ Found: Bagged Trees (bt_model)")
except NameError:
    print("⚠️  Not found: bt_model (Bagged Trees)")

# Subspace KNN
try:
    models_to_save["SKNN_model.pkl"] = sknn_model
    print("✅ Found: Subspace KNN (sknn_model)")
except NameError:
    print("⚠️  Not found: sknn_model (Subspace KNN)")

# Support Vector Machine (Main model)
try:
    models_to_save["SVM_model.pkl"] = svm_model
    print("✅ Found: SVM (svm_model)")
except NameError:
    print("⚠️  Not found: svm_model (SVM)")

# Additional SVM variants if they exist
try:
    models_to_save["SVM_Linear_model.pkl"] = svm_linear
    print("✅ Found: SVM Linear (svm_linear)")
except NameError:
    pass

try:
    models_to_save["SVM_Quadratic_model.pkl"] = svm_quadratic
    print("✅ Found: SVM Quadratic (svm_quadratic)")
except NameError:
    pass

try:
    models_to_save["SVM_Cubic_model.pkl"] = svm_cubic
    print("✅ Found: SVM Cubic (svm_cubic)")
except NameError:
    pass

try:
    models_to_save["SVM_Gaussian_model.pkl"] = svm_gaussian
    print("✅ Found: SVM Gaussian (svm_gaussian)")
except NameError:
    pass

print("\n" + "=" * 60)
print(f"SAVING {len(models_to_save)} MODELS...")
print("=" * 60 + "\n")

# Save each model
saved_count = 0
for filename, model in models_to_save.items():
    try:
        filepath = os.path.join(model_dir, filename)
        joblib.dump(model, filepath)
        print(f"✅ Saved: {filename}")
        saved_count += 1
    except Exception as e:
        print(f"❌ Error saving {filename}: {str(e)}")

print("\n" + "=" * 60)
print(f"🎉 SUCCESS! Saved {saved_count} models")
print(f"📁 Location: {model_dir}")
print("=" * 60)

# Verify the saved files
print("\n📋 Saved model files:")
for f in os.listdir(model_dir):
    if f.endswith('.pkl'):
        filepath = os.path.join(model_dir, f)
        size_kb = os.path.getsize(filepath) / 1024
        print(f"   • {f} ({size_kb:.2f} KB)")

print("\n✅ Models are ready to use with the GUI application!")
print("   Run: streamlit run gui_app/spectral-lungsound-gui.py")
