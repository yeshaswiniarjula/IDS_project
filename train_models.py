
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA

# LOAD DATA
train = pd.read_parquet("KDDTrain.parquet")
test  = pd.read_parquet("KDDTest.parquet")

# ENCODING
encoders = {}
for col in ['protocol_type', 'service', 'flag']:
    le = LabelEncoder()
    le.fit(pd.concat([train[col], test[col]]))
    
    train[col] = le.transform(train[col])
    test[col]  = le.transform(test[col])
    
    encoders[col] = le

# SAVE encoders
joblib.dump(encoders, "label_encoders.joblib")

# PREPARE DATA
drop_cols = ['class', 'classnum']
X_train = train.drop(columns=drop_cols)
X_test  = test.drop(columns=drop_cols)

# SCALING
iso_scaler = StandardScaler()
X_train_iso = iso_scaler.fit_transform(X_train)

svm_scaler = StandardScaler()
X_train_svm = svm_scaler.fit_transform(X_train)

# SAVE scalers
joblib.dump(iso_scaler, "iso_scaler.joblib")
joblib.dump(svm_scaler, "svm_scaler.joblib")

# PCA for SVM
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_svm)

joblib.dump(pca, "svm_pca.joblib")

# MODELS
iso_model = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
iso_model.fit(X_train_iso)

svm_model = OneClassSVM(nu=0.1)
svm_model.fit(X_train_pca)

# SAVE models
joblib.dump(iso_model, "iso_model.joblib")
joblib.dump(svm_model, "svm_model.joblib")

print("✅ All models and preprocessors saved successfully!")
