
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import shutil
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid

# Import metrik lengkap
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ==========================================
# 1. KONFIGURASI DAGSHUB (WAJIB DIISI)
# ==========================================
# Ganti dengan Username & Nama Repo DagsHub kamu
DAGSHUB_REPO_OWNER = "AndreasM004D5Y0218"
DAGSHUB_REPO_NAME = "heart_disease_uci_prediction"

try:
    dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)
    mlflow.set_experiment("Heart_Disease_Prediction_Complete")
    print("[INFO] DagsHub Connected!")
except Exception as e:
    print(f"[ERROR] Koneksi DagsHub Gagal: {e}")

# ==========================================
# 2. LOAD DATA
# ==========================================
# Path disesuaikan dengan folder heart_disease_uci_preprocessing
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, 'heart_disease_uci_preprocessing')

train_path = os.path.join(data_dir, 'train_data.csv')
test_path = os.path.join(data_dir, 'test_data.csv')

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Pisahkan Fitur & Target (Target kita adalah 'num')
X_train = train_df.drop("num", axis=1)
y_train = train_df["num"]
X_test = test_df.drop("num", axis=1)
y_test = test_df["num"]

# Contoh input untuk MLflow signature
input_example = X_train.iloc[:5]

# ==========================================
# 3. DEFINISI PARAMETER (HYPERPARAMETER SPACE)
# ==========================================
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 10],
    "min_samples_split": [2, 5]
}

# ==========================================
# 4. LOOPING MANUAL
# ==========================================
print(f"Mulai Training dengan {len(list(ParameterGrid(param_grid)))} kombinasi parameter...")

for i, params in enumerate(ParameterGrid(param_grid)):

    # Nama run dinamis biar rapi di Dashboard
    nama_run = f"Run_{i+1}_Est-{params['n_estimators']}_Depth-{params['max_depth']}"

    with mlflow.start_run(run_name=nama_run):
        print(f"Running: {nama_run} ...")

        # A. Train
        model = RandomForestClassifier(random_state=42, **params)
        model.fit(X_train, y_train)

        # B. Predict
        y_pred = model.predict(X_test)

        # C. Hitung SEMUA Metrik (Weighted karena target multiclass 0-4)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # D. Log ke DagsHub
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # E. Log Artifacts (Gambar & Laporan)
        # Buat folder sementara
        os.makedirs("temp_artifacts", exist_ok=True)

        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"CM - Acc: {acc:.2f}")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')

        cm_path = "temp_artifacts/confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)

        # 2. HTML Report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        html_content = f"<html><body><h3>Classification Report</h3><pre>{json.dumps(report, indent=2)}</pre></body></html>"

        report_path = "temp_artifacts/classification_report.html"
        with open(report_path, "w") as f:
            f.write(html_content)
        mlflow.log_artifact(report_path)

        # 3. Model & Schema
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

        # Bersih-bersih folder temp setelah upload selesai
        if os.path.exists("temp_artifacts"):
            shutil.rmtree("temp_artifacts")

print("\n[SELESAI] Semua eksperimen telah diupload ke DagsHub!")
