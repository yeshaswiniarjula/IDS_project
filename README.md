\# SentinelNet IDS



SentinelNet IDS is a machine learning-based Intrusion Detection System built as a mini project:

\*\*"SentinelNet — Smart Network Intrusion Detection System."\*\*



This version implements the core modules end-to-end:



\- Network traffic classification using supervised ML (SVM with PCA)

\- Anomaly detection using unsupervised ML (Isolation Forest)

\- Synthetic data generation using CTGAN for class balancing

\- Interactive Streamlit dashboard for real-time predictions

\- Trained models saved and loaded via joblib for fast inference

\- KDD Cup dataset (train/test) for benchmarking



\---



\## Tech Stack



\- Python 3.x

\- Streamlit

\- Scikit-learn (SVM, Isolation Forest, PCA)

\- CTGAN (synthetic data generation)

\- Pandas, NumPy

\- Matplotlib, Seaborn

\- Joblib (model persistence)

\- PyArrow (parquet file support)



\---



\## Project Structure



```

IDS\_project/

├── appli.py                         # Streamlit web application

├── train\_models.py                  # Script to train SVM and Isolation Forest models

├── convert.py                       # Data conversion utility (CSV to Parquet)

├── sentinelNet.ipynb                # Main ML notebook (supervised approach)

├── Unsupervised(sentinelNet).ipynb  # Unsupervised/anomaly detection notebook

├── KDDTrain.parquet                 # Training dataset (KDD Cup)

├── KDDTest.csv                      # Test dataset (KDD Cup)

├── KDDTest.parquet                  # Test dataset (Parquet format)

├── ctgan\_synthetic\_data.csv         # CTGAN-generated synthetic samples

├── final\_attack\_predictions.csv     # Output predictions file

├── svm\_model.joblib                 # Trained SVM model

├── svm\_scaler.joblib                # Scaler for SVM input features

├── svm\_pca.joblib                   # PCA transformer for SVM

├── iso\_model.joblib                 # Trained Isolation Forest model

├── iso\_scaler.joblib                # Scaler for Isolation Forest

├── label\_encoders.joblib            # Label encoders for categorical features

├── requirements.txt                 # Python dependencies

├── runtime.txt                      # Python runtime version

└── .streamlit/                      # Streamlit configuration

```



\---



\## Run Locally



```bash

\# 1. Clone the repository

git clone https://github.com/yeshaswiniarjula/IDS\_project.git

cd IDS\_project



\# 2. Create a virtual environment

python3 -m venv .venv

source .venv/bin/activate        # On Windows: .venv\\Scripts\\activate



\# 3. Install dependencies

pip install -r requirements.txt



\# 4. Run the Streamlit app

streamlit run appli.py

```



Then open `http://localhost:8501` in your browser.



\---



\## Train Models (Optional)



If you want to retrain the models from scratch:



```bash

python3 train\_models.py

```



This will regenerate the following saved model files:

\- `svm\_model.joblib`

\- `svm\_scaler.joblib`

\- `svm\_pca.joblib`

\- `iso\_model.joblib`

\- `iso\_scaler.joblib`

\- `label\_encoders.joblib`



Once models exist, the Streamlit app auto-loads them for predictions.



\---



\## ML Models



\### Supervised — SVM Classifier

\- Algorithm: Support Vector Machine (SVM)

\- Preprocessing: Label Encoding → Standard Scaling → PCA

\- Task: Multi-class attack type classification

\- Dataset: KDD Cup 99 (Train/Test)



\### Unsupervised — Isolation Forest

\- Algorithm: Isolation Forest

\- Task: Anomaly detection (normal vs. attack traffic)

\- Used when labeled data is unavailable



\### Synthetic Data — CTGAN

\- Used to generate synthetic network traffic samples

\- Helps balance minority attack classes in training data



\---



\## Dataset



This project uses the \*\*KDD Cup 1999\*\* dataset — a standard benchmark for intrusion detection research.



\- `KDDTrain.parquet` — Training split

\- `KDDTest.csv` / `KDDTest.parquet` — Test split



Features include network connection attributes such as protocol type, service, flag, byte counts, and more.



\---



\## About



SentinelNet IDS is a machine learning-based intrusion detection system that identifies malicious network activity using both supervised (SVM) and unsupervised (Isolation Forest) approaches, with a Streamlit-based interactive interface for visualization and prediction.

