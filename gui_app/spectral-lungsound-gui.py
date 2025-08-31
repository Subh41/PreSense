import streamlit as st
import numpy as np
import pandas as pd
import joblib, os, time
import librosa, librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import rfft, rfftfreq
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import plotly.express as px
import time

# -----------------------------
# PAGE CONFIG & THEME
# -----------------------------
st.set_page_config(
    page_title="Lung Sound Classifier",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for pro styling
st.markdown("""
<style>
    .reportview-container {background: linear-gradient(135deg, #eef2f3, #8e9eab);}
    .sidebar .sidebar-content {background: #2c3e50; color: white;}
    h1,h2,h3 {color: #2c3e50;}
    .stSuccess {background-color: #27ae60; color:white;}
    .stError {background-color: #c0392b; color:white;}
    .stWarning {background-color: #f39c12; color:white;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD TRAINED MODELS
# -----------------------------
model_dir = r"D:\Notebooks\Spectral-paper-Implementation\Trained-models"
models = {}
for f in os.listdir(model_dir):
    if f.endswith(".pkl"):
        name = f.replace("_model.pkl","")
        models[name] = joblib.load(os.path.join(model_dir,f))

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
page = st.sidebar.radio("📑 Navigation", ["🏠 Home", "📂 Upload & Preprocess", "📊 Features & Models", "📈 Results Dashboard", "ℹ️ About"])


# -----------------------------
# HOME PAGE
# -----------------------------
if page == "🏠 Home":
    st.markdown("<h1 style='text-align:center; color:#2c3e50;'>🫁 Lung Sound Classification System</h1>", unsafe_allow_html=True)

    # Logos centered in one row
    col1, col2, col3 = st.columns([1,2,1])
    with col1:
        st.image("Downloads/channels4_profile.jpg", caption="ResearchPedia", width=180)
    with col2:
        st.markdown("<h3 style='text-align:center; color:#34495e;'>Powered by</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div style="text-align:center; font-size:18px; color:#2c3e50;">
            <span style="color:#2980b9; font-weight:bold;">Community of Research and Development (CRD)</span><br>
            & <span style="color:#d35400; font-weight:bold;">ResearchPedia</span><br><br>
            👩‍💻 <span style="color:#16a085; font-weight:bold;">Developed by Engr. Misha Urooj Khan</span>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.image("Downloads/Misha-Urooj-Khan.webp", caption="CRD", width=180)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Welcome message with styled box
    st.markdown("""
    <div style="background-color:#ecf0f1; padding:20px; border-radius:10px;">
        <p style="font-size:16px; color:#2c3e50; text-align:justify;">
        Welcome to the <b>Lung Sound Classifier</b>, a <b style="color:#2980b9;">professional biomedical tool</b> that 
        analyzes lung sounds and classifies them into <b>Normal</b>, <b>Asthma</b>, or <b>Pneumonia</b> 
        using advanced <b>Machine Learning models</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Feature List
    st.markdown("""
    <h3 style="color:#2c3e50;">🚀 Key Features:</h3>
    <ul style="font-size:16px; line-height:1.6; color:#2c3e50;">
        <li>Preprocessing with <b>normalization, filtering, silence removal</b></li>
        <li>Extraction of <b>9+ IEEE spectral features</b></li>
        <li>Evaluation across multiple trained models (<b>SVM, KNN, Trees, Naïve Bayes</b>, etc.)</li>
        <li>Confusion matrices & detailed performance metrics</li>
        <li>Final classification with <b>probability scores</b></li>
        <li>Option to <b>download results as a professional report</b></li>
    </ul>
    """, unsafe_allow_html=True)



# -----------------------------
# UPLOAD & PREPROCESS
# -----------------------------
elif page == "📂 Upload & Preprocess":
    st.header("📂 Upload Lung Sound Recording")
    uploaded_file = st.file_uploader("Upload a lung sound file (.wav)", type=["wav"])
    
    if uploaded_file is not None:
        import soundfile as sf
        y, sr = librosa.load(uploaded_file, sr=44100, mono=True)
        st.audio(uploaded_file, format="audio/wav")

        st.subheader("🔧 Preprocessing")
        start = time.time()

        # Normalize + Filter
        y_norm = (y - np.min(y)) / (np.ptp(y)+1e-12)
        from scipy.signal import butter, sosfiltfilt
        sos = butter(10, [250,2000], btype="bandpass", fs=sr, output="sos")
        y_filt = sosfiltfilt(sos, y_norm)

        # Plots: Raw vs Filtered
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(7,3))
            librosa.display.waveshow(y, sr=sr, ax=ax, color="royalblue")
            ax.set_title("Raw Waveform")
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots(figsize=(7,3))
            librosa.display.waveshow(y_filt, sr=sr, ax=ax, color="darkorange")
            ax.set_title("Filtered (250–2000 Hz)")
            st.pyplot(fig)

        st.success(f"✅ Preprocessing completed in {time.time()-start:.2f} sec")
        st.session_state["signal"] = y_filt
        st.session_state["sr"] = sr

# -----------------------------
# FEATURE EXTRACTION & MODELS
# -----------------------------
elif page == "📊 Features & Models":
    if "signal" not in st.session_state:
        st.warning("⚠️ Please upload and preprocess a file first.")
    else:
        y_filt, sr = st.session_state["signal"], st.session_state["sr"]
        st.header("📊 Feature Extraction")
        start = time.time()

        # Compute FFT
        X = np.abs(rfft(y_filt))
        F = rfftfreq(len(y_filt), 1/sr)

        # Extract 9 IEEE Features
        SCN = np.sum(F*X)/np.sum(X)
        SCR = np.max(X)/np.mean(X)
        SDC = np.sum((X[1:]-X[0])/np.arange(1,len(X)))/np.sum(X)
        P = X/np.sum(X)
        SEN = -np.sum(P*np.log2(P+1e-12))
        SFL = np.exp(np.mean(np.log(X+1e-12)))/(np.mean(X)+1e-12)
        SFLUX = np.sum(np.diff(X)**2)/len(X)
        total_energy = np.sum(X)
        threshold = 0.85*total_energy
        SRO = F[np.where(np.cumsum(X)>=threshold)[0][0]]
        SSL = np.polyfit(F, X, 1)[0]
        SSP = np.sqrt(np.sum(((F-SCN)**2)*X)/np.sum(X))

        feature_vector = np.array([[SCN,SCR,SDC,SEN,SFL,SFLUX,SRO,SSL,SSP]])
        feat_names = ["SCN","SCR","SDC","SEN","SFL","SFLUX","SRO","SSL","SSP"]

        # Interactive Bar Plot with Plotly
        df_feat = pd.DataFrame({"Feature": feat_names, "Value": feature_vector[0]})
        fig = px.bar(
            df_feat, 
            x="Feature", y="Value", 
            title="Extracted Features",
            color="Value", 
            color_continuous_scale="viridis"
        )
        fig.update_layout(
            xaxis_title="Features",
            yaxis_title="Value",
            template="plotly_white"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.success(f"✅ Feature extraction completed in {time.time()-start:.2f} sec")
        st.session_state["features"] = feature_vector


# -----------------------------
# RESULTS DASHBOARD
# -----------------------------
elif page == "📈 Results Dashboard":
    if "features" not in st.session_state:
        st.warning("⚠️ Please extract features first.")
    else:
        st.header("📈 Model Evaluation Dashboard")
        X_feat = st.session_state["features"]

        results = []
        for name, model in models.items():
            # Measure prediction time
            start = time.time()
            y_pred = model.predict(X_feat)[0]
            pred_time = time.time() - start

            results.append({
                "Model": name,
                "Prediction": y_pred,
                "Prediction Time (s)": f"{pred_time:.4f}"
            })

        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)

        # Final Prediction (majority vote)
        final_pred = results_df["Prediction"].mode()[0]
        st.subheader("🏆 Final Classification")
        st.success(f"🫁 This lung sound is classified as **{final_pred}**")

        # -----------------------------
        # Download Results as Report
        # -----------------------------
        st.markdown("### 📥 Download Report")
        report_text = f"""
        🫁 Lung Sound Classification Report
        ==================================

        ✅ Final Prediction: {final_pred}

        📊 Model Predictions & Accuracies:
        {results_df.to_string(index=False)}

        ⚙️ Extracted Features:
        {dict(zip(['SCN','SCR','SDC','SEN','SFL','SFLUX','SRO','SSL','SSP'], 
                  st.session_state['features'][0]))}
        """

        st.download_button(
            label="⬇️ Download TXT Report",
            data=report_text,
            file_name="lung_sound_report.txt",
            mime="text/plain"
        )

        st.download_button(
            label="⬇️ Download CSV Results",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="model_predictions.csv",
            mime="text/csv"
        )


# -----------------------------
# ABOUT PAGE
# -----------------------------
elif page == "ℹ️ About":
    st.header("ℹ️ About this App")
    st.markdown("""
    **Lung Sound Classifier v1.0**  
    Developed as a professional biomedical software interface for research & teaching.  

    - Built with **Streamlit + Scikit-learn + Librosa**  
    - Implements IEEE spectral features  
    - Supports multiple models (LDA, KNN, NB, Trees, SVM)  
    - Ready for deployment in labs, hospitals, or e-learning.  
    """)
