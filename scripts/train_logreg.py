import os
import glob
import numpy as np
import pandas as pd
import kagglehub
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score,
    matthews_corrcoef
)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET_COL = "disease"
TEST_SIZE = 0.20
RANDOM_STATE = 42

SUBSAMPLE_N = 500_000  

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

if SUBSAMPLE_N is not None and len(df) > SUBSAMPLE_N:
    df = df.sample(n=SUBSAMPLE_N, random_state=RANDOM_STATE)
    print(f"\nUsing subsample for Logistic Regression: {SUBSAMPLE_N} rows")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop"
)

clf = LogisticRegression(
    solver="lbfgs",
    max_iter=6000,
    class_weight="balanced"
)

model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("clf", clf)
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
mcc = matthews_corrcoef(y_test, y_pred)

print("\n=== Logistic Regression Results (India Waterborne Disease) ===")
print(f"Target: {TARGET_COL}")
print(f"Classes: {list(model.named_steps['clf'].classes_)}")
print(f"Accuracy : {acc:.4f}")
print(f"AUC      : {auc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"MCC      : {mcc:.4f}")

model_path = os.path.join(MODEL_DIR, "logistic_regression.pkl")
joblib.dump(model, model_path, compress=3)
print(f"\n✅ Saved model to: {model_path}")

split_path = os.path.join(MODEL_DIR, "test_split.pkl")
joblib.dump((X_test, y_test), split_path, compress=3)
print(f"✅ Saved test split to: {split_path}")
