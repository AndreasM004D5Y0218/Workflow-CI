
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# 1. Load Data
# Mengarah ke folder yang barusan kamu buat
train_df = pd.read_csv('heart_disease_uci_preprocessing/train_data.csv')
test_df = pd.read_csv('heart_disease_uci_preprocessing/test_data.csv')

# 2. Pisahkan Fitur dan Target
# Di Heart Disease UCI, targetnya adalah kolom 'num' (0=sehat, 1-4=sakit)
# Jika di preprocessing kamu mengubah namanya jadi 'target', ganti 'num' di bawah jadi 'target'
target_col = 'num'

X_train = train_df.drop(target_col, axis=1)
y_train = train_df[target_col]
X_test = test_df.drop(target_col, axis=1)
y_test = test_df[target_col]

# 3. MLflow BASIC (Lokal)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("heart_disease_autolog")
mlflow.sklearn.autolog()

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Trained. Accuracy: {acc:.4f}")
