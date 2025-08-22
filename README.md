Spectral Analysis of Lung Sounds for Asthma & Pneumonia Classification
<p align="center"> <img src="results/figures/methodology.png" width="600"><br> <em>Figure 1. Proposed Methodology (from paper)</em> </p>
📖 Overview

This repository contains all resources related to my IEEE conference paper:

“Spectral Analysis of Lungs sounds for Classification of Asthma and Pneumonia Wheezing” (Proc. ICECCE 2020, Istanbul, Turkey)

Lung sounds provide critical diagnostic information. Wheezing is a key indicator of pulmonary illnesses such as Asthma and Pneumonia. This project presents a complete machine learning pipeline — from preprocessing raw lung sounds, to spectral feature extraction, and classification via Support Vector Machine (SVM) with an accuracy of 96.7%.

📂 Repository Contents

paper/ → Full IEEE paper (PDF)

notebooks/ → Kaggle notebooks for preprocessing, feature extraction & classification

gui_app/ → Ready-to-deploy GUI for lung sound classification

data/ → Example lung sound samples (optional small demo files)

results/figures/ → Figures and plots from preprocessing & classification

⚙️ Environment Setup

We recommend using Anaconda.

# Create environment
conda env create -f environment.yml
conda activate lungsound-env


Key libraries: numpy, scipy, librosa, matplotlib, scikit-learn, tkinter.

🛠️ How to Run
1. Preprocessing Notebook

Open notebooks/preprocessing.ipynb in Kaggle/Jupyter.
It demonstrates:

Normalization (Min-Max scaling)

Segmentation (250 ms frames)

Butterworth bandpass filtering (250 Hz – 2 kHz)

<p align="center"> <img src="results/figures/preprocessing.png" width="600"><br> <em>Figure 2. Preprocessing of Lung Sound (time & frequency domain)</em> </p>
2. Feature Extraction

We extract 9 spectral features from lung sounds:

Spectral Centroid

Spectral Crest

Spectral Decrease

Spectral Entropy

Spectral Flatness

Spectral Flux

Spectral Roll-off

Spectral Slope

Spectral Spread

<p align="center"> <img src="results/figures/features.png" width="600"><br> <em>Figure 3. Scatter plots of spectral features showing class separation</em> </p>
3. Classification

Classifiers tested: LD, KNN, Decision Trees, Naïve Bayes, Bagged Trees, Subspace KNN.

Linear SVM achieved 96.7% accuracy with 5-fold cross validation.

<p align="center"> <img src="results/figures/svm_accuracy.png" width="500"><br> <em>Figure 4. Accuracy of SVM kernels</em> </p>
4. GUI Application 🎛️

Run the GUI app to classify any .wav lung sound file:

python gui_app/lungsound_gui.py


Features:

Load lung sound file

View preprocessing plots + time taken

Extract features & visualize them

Run through all trained models

Display final classification (Asthma / Pneumonia / Normal) with accuracy, precision, recall

<p align="center"> <img src="results/figures/gui_demo.png" width="500"><br> <em>Figure 5. GUI interface for Lung Sound Classification</em> </p>
📊 Results

Accuracy: 96.7% (Linear SVM)

Cross-validation: 5-fold (96.7%), 10-fold (96.4%)

Confusion matrix shows strong separation between classes

<p align="center"> <img src="results/figures/confusion_matrix.png" width="500"><br> <em>Figure 6. Confusion Matrix</em> </p>
📜 Reference

If you use this work, please cite:
M. Arooj, S.Z.H. Naqvi, M.U. Khan, M.A. Choudhary, S. Aziz, M.N. Hassan,
“Spectral Analysis of Lungs sounds for Classification of Asthma and Pneumonia Wheezing,”
Proc. of 2nd Int. Conf. Electrical, Communication and Computer Engineering (ICECCE), 2020.

✨ This repository is designed as an educational resource for students and researchers interested in Biomedical Signal Processing & Machine Learning for healthcare.
