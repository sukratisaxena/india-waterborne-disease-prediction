import os
import glob
import numpy as np
import pandas as pd
import kagglehub
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score,
    matthews_corrcoef
)

from xgboost import XGBClassifier


MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "xgboost.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

TEST_SPLIT_SHARED_PATH = os.path.join(MODEL_DIR, "test_split.pkl")

TEST_SPLIT_XGB_ENCODED_PATH = os.path.join(MODEL_DIR, "test_split_xgb_encoded.pkl")

TARGET_COL = "disease"

USE_SUBSAMPLE = True
SUBSAMPLE_N = 500_000


def make_onehot_sparse():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


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

    if USE_SUBSAMPLE and len(df) > SUBSAMPLE_N:
        df = df.sample(n=SUBSAMPLE_N, random_state=42)
        print(f"\nUsing subsample for XGBoost: {SUBSAMPLE_N} rows")

    X = df.drop(columns=[TARGET_COL])
    y_str = df[TARGET_COL].values  

    le = LabelEncoder()
    y_enc = le.fit_transform(y_str)
    class_names = list(le.classes_)
    print("\nEncoded classes:", {i: name for i, name in enumerate(class_names)})

    X_train, X_test, y_train_enc, y_test_enc, y_train_str, y_test_str = train_test_split(
        X, y_enc, y_str,
        test_size=0.20,
        random_state=42,
        stratify=y_enc
    )

    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_onehot_sparse())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )

    num_class = len(class_names)

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=num_class,
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42
    )

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", clf)
    ])

    model.fit(X_train, y_train_enc)

    y_pred_enc = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    y_pred_str = le.inverse_transform(y_pred_enc.astype(int))

    acc = accuracy_score(y_test_str, y_pred_str)
    prec = precision_score(y_test_str, y_pred_str, average="weighted", zero_division=0)
    rec = recall_score(y_test_str, y_pred_str, average="weighted", zero_division=0)
    f1 = f1_score(y_test_str, y_pred_str, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_test_str, y_pred_str)

    auc = roc_auc_score(
        y_test_str,
        y_proba,
        multi_class="ovr",
        average="weighted",
        labels=class_names
    )

    print("\n=== XGBoost Results (India Waterborne Disease) ===")
    print("Target:", TARGET_COL)
    print("Classes:", class_names)
    print(f"Accuracy : {acc:.4f}")
    print(f"AUC      : {auc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"MCC      : {mcc:.4f}")

    joblib.dump(model, MODEL_PATH)
    print(f"\n✅ Saved model to: {MODEL_PATH}")

    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f"✅ Saved label encoder to: {LABEL_ENCODER_PATH}")

    joblib.dump((X_test, y_test_str), TEST_SPLIT_SHARED_PATH)
    print(f"✅ Saved SHARED test split (string labels) to: {TEST_SPLIT_SHARED_PATH}")

    joblib.dump((X_test, y_test_enc), TEST_SPLIT_XGB_ENCODED_PATH)
    print(f"✅ Saved XGB encoded test split to: {TEST_SPLIT_XGB_ENCODED_PATH}")


if __name__ == "__main__":
    main()
