import os
import glob
import numpy as np
import pandas as pd
import kagglehub
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score,
    matthews_corrcoef
)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET_COL = "disease"
USE_SUBSAMPLE = True
SUBSAMPLE_N = 120_000   

TEST_SIZE = 0.20
RANDOM_STATE = 42

MODEL_PATH = os.path.join(MODEL_DIR, "knn.pkl")
TEST_SPLIT_PATH = os.path.join(MODEL_DIR, "test_split.pkl")

def main():
    dataset_path = kagglehub.dataset_download("bhargavadepu/india-waterborne-disease-dataset")
    print("Path to dataset files:", dataset_path)

    csv_files = glob.glob(os.path.join(dataset_path, "**", "*.csv"), recursive=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found inside: {dataset_path}")

    csv_path = max(csv_files, key=os.path.getsize)
    print("Using CSV:", csv_path)

    df = pd.read_csv(csv_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Columns: {list(df.columns)}")

    df = df.dropna(subset=[TARGET_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()

    print("\nTarget distribution (top 20):")
    print(df[TARGET_COL].value_counts().head(20))

    id_like = [c for c in df.columns if c.lower() in {"id", "patient_id", "record_id"} or c.lower().endswith("_id")]
    df = df.drop(columns=id_like, errors="ignore")

    X_full = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]


    numeric_cols = X_full.select_dtypes(include=["int64", "int32", "float64", "float32", "bool"]).columns.tolist()

    if not numeric_cols:
        raise ValueError("No numeric/bool columns found. KNN needs numeric inputs.")

    X = X_full[numeric_cols].copy()

    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)

    print(f"\nKNN using ONLY numeric/bool features: {len(numeric_cols)} columns")
    print("Example columns:", numeric_cols[:15])

    if USE_SUBSAMPLE and len(df) > SUBSAMPLE_N:
        df_small = pd.concat([X, y], axis=1).sample(n=SUBSAMPLE_N, random_state=RANDOM_STATE)
        X = df_small.drop(columns=[TARGET_COL])
        y = df_small[TARGET_COL]
        print(f"\nUsing subsample for KNN: {SUBSAMPLE_N} rows")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(
            n_neighbors=15,
            weights="distance",
            algorithm="auto"
        ))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")

    print("\n=== KNN Results (numeric-only for Streamlit) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"AUC      : {auc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"MCC      : {mcc:.4f}")

    joblib.dump(model, MODEL_PATH, compress=3)
    print(f"\n✅ Saved model to: {MODEL_PATH}")

    joblib.dump((X_test, y_test), TEST_SPLIT_PATH, compress=3)
    print(f"✅ Saved test split to: {TEST_SPLIT_PATH}")

if __name__ == "__main__":
    main()
