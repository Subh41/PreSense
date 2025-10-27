from setuptools import setup, find_packages

setup(
    name="lungeval",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.0',
        'scipy>=1.6.0',
        'librosa>=0.8.1',
        'matplotlib>=3.3.0',
        'scikit-learn>=0.24.0',
        'pandas>=1.2.0',
        'seaborn>=0.11.0',
        'plotly>=5.0.0',
        'joblib>=1.0.0',
        'soundfile>=0.10.0',
        'streamlit>=1.0.0',
        'openpyxl>=3.0.0',
        'tqdm>=4.60.0',
        'streamlit-extras>=1.0.0',
        'numba>=0.58.0',
        'llvmlite>=0.41.0',
    ],
)
