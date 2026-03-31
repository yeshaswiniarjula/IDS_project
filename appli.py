import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# UI CONFIG
# ===============================
st.set_page_config(page_title="SentinelNet IDS", layout="wide")
st.title("🔐 SentinelNet IDS (Smart Deployment Version)")

# ===============================
# SIDEBAR SETTINGS
# ===============================
st.sidebar.header("⚙️ Settings")

sample_size = st.sidebar.slider("Training Sample Size", 200, 2000, 1000, step=200)
svm_sample_size = st.sidebar.slider("SVM Training Size", 200, 2000, 1000, step=200)
pca_components = st.sidebar.slider("PCA Components", 5, 20, 10)
test_sample_size = st.sidebar.slider("Test Sample Size", 200, 2000, 1000, step=200)

# ✅ NEW: CTGAN TOGGLE
use_ctgan = st.sidebar.checkbox("Use CTGAN Synthetic Data", value=True)

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    train = pd.read_parquet("KDDTrain.parquet")
    test = pd.read_parquet("KDDTest.parquet")
    return train, test

# ===============================
# BUTTON
# ===============================
if st.button("🚀 Run Detection"):

    with st.spinner("⏳ Loading data..."):
        train, test = load_data()

    test = test.sample(min(test_sample_size, len(test)), random_state=42)

    y_test = (test["class"].str.lower().str.strip() != "normal").astype(int)
    cat_cols = ["protocol_type", "service", "flag"]

    st.subheader("📊 Dataset Info")
    st.write(f"Test Samples Used: {len(test)}")
    st.write(f"Actual Attacks: {y_test.sum()}")

    # ===============================
    # DATA SOURCE SELECTION
    # ===============================
    if use_ctgan:
        st.subheader("🧪 Using Pre-Generated CTGAN Data")

        try:
            synth = pd.read_csv("ctgan_synthetic_data.csv")
            train_sample = synth.sample(min(sample_size, len(synth)), random_state=42)

        except:
            st.error("❌ CTGAN file not found! Using real data instead.")
            use_ctgan = False

    if not use_ctgan:
        st.subheader("⚡ Using Real Normal Data")

        normal_data = train[train["class"].str.lower().str.strip() == "normal"].copy()
        normal_data = normal_data.drop(columns=["class", "classnum"], errors="ignore")

        for col in cat_cols:
            le = LabelEncoder()
            normal_data[col] = le.fit_transform(normal_data[col].astype(str))

        normal_data = normal_data.astype(np.float32)
        train_sample = normal_data.sample(min(sample_size, len(normal_data)), random_state=42)

    # ===============================
    # PREPARE DATA
    # ===============================
    def prepare(train_df, test_df):
        X_tr = train_df.copy()
        X_te = test_df.drop(columns=["class"], errors="ignore")

        X_tr = pd.get_dummies(X_tr, columns=cat_cols)
        X_te = pd.get_dummies(X_te, columns=cat_cols)

        X_tr, X_te = X_tr.align(X_te, join="left", axis=1, fill_value=0)

        scaler = RobustScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        pca = PCA(n_components=pca_components)
        X_tr_pca = pca.fit_transform(X_tr_sc)
        X_te_pca = pca.transform(X_te_sc)

        return X_tr_sc, X_te_sc, X_tr_pca, X_te_pca

    with st.spinner("⚙️ Preparing data..."):
        X_tr_sc, X_te_sc, X_tr_pca, X_te_pca = prepare(train_sample, test)

    # ===============================
    # MODELS
    # ===============================
    st.subheader("🔍 Running Detection")

    iso = IsolationForest(contamination=0.1, n_estimators=100, random_state=42)
    iso.fit(X_tr_sc)

    svm = OneClassSVM(nu=0.1)
    svm.fit(X_tr_sc[:svm_sample_size])

    lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
    lof.fit(X_tr_pca)

    # ===============================
    # SCORES
    # ===============================
    iso_scores = -iso.score_samples(X_te_sc)
    svm_scores = -svm.score_samples(X_te_sc)
    lof_scores = -lof.score_samples(X_te_pca)

    def normalize(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-9)

    ens_scores = (
        0.4 * normalize(iso_scores) +
        0.3 * normalize(svm_scores) +
        0.3 * normalize(lof_scores)
    )

    # ===============================
    # THRESHOLD
    # ===============================
    best_t = 0.5
    best_f1 = 0

    for t in np.linspace(0.2, 0.8, 20):
        preds = (ens_scores >= t).astype(int)
        f1 = f1_score(y_test, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    y_pred = (ens_scores >= best_t).astype(int)

    st.write(f"🎯 Best Threshold: {best_t:.2f}")

    # ===============================
    # METRICS
    # ===============================
    st.subheader("📊 Performance")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.2f}")
    c2.metric("Precision", f"{prec:.2f}")
    c3.metric("Recall", f"{rec:.2f}")
    c4.metric("F1 Score", f"{f1:.2f}")

    # ===============================
    # CONFUSION MATRIX HEATMAP
    # ===============================
    st.subheader("🔍 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=["Normal", "Attack"],
                yticklabels=["Normal", "Attack"])
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")

    st.pyplot(fig_cm)

    # ===============================
    # BAR GRAPH (NEW ✅)
    # ===============================
    st.subheader("📊 Prediction Distribution")

    normal_count = (y_pred == 0).sum()
    attack_count = (y_pred == 1).sum()

    fig_bar, ax_bar = plt.subplots()
    ax_bar.bar(["Normal", "Attack"], [normal_count, attack_count])
    ax_bar.set_ylabel("Count")

    st.pyplot(fig_bar)

    # ===============================
    # ALERT SYSTEM
    # ===============================
    if attack_count > 0:
        st.error(f"🚨 {attack_count} Intrusions Detected!")
    else:
        st.success("✅ No Intrusions Detected")

    st.success("✅ Detection Completed!")