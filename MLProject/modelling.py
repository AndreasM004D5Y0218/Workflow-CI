import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# --- Hyperparameters Setup ---
n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10
min_samples_split = int(sys.argv[3]) if len(sys.argv) > 3 else 2
if max_depth == 0:
    max_depth = None

# --- Data Loading ---
train_path = "heart_disease_uci_preprocessing/train_data.csv"
test_path = "heart_disease_uci_preprocessing/test_data.csv"

if not os.path.exists(train_path):
    sys.exit(f"Error: Dataset not found at {train_path}")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_col = 'num'
X_train = train_df.drop(target_col, axis=1)
y_train = train_df[target_col]
X_test = test_df.drop(target_col, axis=1)
y_test = test_df[target_col]

input_example = X_train.iloc[:5]

# --- MLflow Execution ---
with mlflow.start_run():
    # 1. Train Model
    model = RandomForestClassifier(
        n_estimators=n_estimators, 
        max_depth=max_depth, 
        min_samples_split=min_samples_split,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 2. Calculate Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f"Training finished. Accuracy: {acc:.4f}")

    # 3. Log to MLflow
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("min_samples_split", min_samples_split)
    
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # 4. Generate & Save Artifacts Locally
    output_dir = "artifacts"
    os.makedirs(output_dir, exist_ok=True)

    # a. Metrics Text
    with open(f"{output_dir}/metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1: {f1:.4f}\n")

    # b. Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix (Acc: {acc:.2f})")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix.png")
    plt.close()

    # c. HTML Classification Report
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    html_content = f"<html><body><h3>Classification Report</h3><pre>{json.dumps(report_dict, indent=2)}</pre></body></html>"
    with open(f"{output_dir}/classification_report.html", "w") as f:
        f.write(html_content)

    # 5. Log Artifacts & Model to MLflow
    mlflow.log_artifacts(output_dir, artifact_path="additional_artifacts")
    
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example
    )
