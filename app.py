import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": "#0E1117",
    "axes.facecolor": "#161B22",
    "axes.edgecolor": "#FAFAFA",
    "axes.labelcolor": "#FAFAFA",
    "text.color": "#FAFAFA",
    "xtick.color": "#FAFAFA",
    "ytick.color": "#FAFAFA",
    "grid.color": "#30363D",
})

st.set_page_config(page_title="Disease Prediction App", layout="wide")

MODEL_DIR = "models"
TARGET_COL = "disease"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}

TEST_SPLIT_PATH = os.path.join(MODEL_DIR, "test_split.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_resource
def load_test_split():
    return joblib.load(TEST_SPLIT_PATH)

@st.cache_resource
def load_label_encoder():
    if os.path.exists(LABEL_ENCODER_PATH):
        return joblib.load(LABEL_ENCODER_PATH)
    return None

def align_X_to_model(X, model):
    if hasattr(model, "feature_names_in_"):
        X = X.copy()
        for c in model.feature_names_in_:
            if c not in X.columns:
                X[c] = np.nan
        return X[model.feature_names_in_]
    return X

def normalize_labels(y, le=None):
    y = np.asarray(y).ravel()
    if le is not None and np.issubdtype(y.dtype, np.number):
        y = le.inverse_transform(y.astype(int))
    return y.astype(str)

def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"),
        "Precision": precision_score(y_true, y_pred, average="weighted"),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "F1": f1_score(y_true, y_pred, average="weighted"),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

def plot_confusion_matrix(cm, class_names):
    fig = plt.figure(figsize=(12, 10))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix", fontsize=22)
    plt.xlabel("Predicted", fontsize=16)
    plt.ylabel("Actual", fontsize=16)
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.yticks(range(len(class_names)), class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    return fig

st.sidebar.header("Controls")

model_name = st.sidebar.selectbox("Select Model", list(MODEL_FILES.keys()))
model = load_model(os.path.join(MODEL_DIR, MODEL_FILES[model_name]))

le = load_label_encoder()
class_names = list(le.classes_) if le else model.classes_

@st.cache_data
def get_test_csv_bytes():
    X_test, y_test = load_test_split()
    df = X_test.copy()
    df[TARGET_COL] = y_test
    return df.to_csv(index=False).encode("utf-8")

st.sidebar.download_button(
    "Download Test Dataset",
    data=get_test_csv_bytes(),
    file_name="test_data.csv",
    mime="text/csv"
)

uploaded_file = st.sidebar.file_uploader("Upload Test Dataset", type=["csv"])

X_eval, y_eval_raw = load_test_split()
source_name = "Default Test Data"

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    y_eval_raw = df[TARGET_COL]
    X_eval = df.drop(columns=[TARGET_COL])
    source_name = "Uploaded Dataset"

X_eval = align_X_to_model(X_eval, model)

st.title("India Waterborne Disease Prediction")

col1, col2, col3 = st.columns(3)
col1.metric("Model", model_name)
col2.metric("Samples", len(X_eval))
col3.metric("Features", len(X_eval.columns))

st.subheader("Dataset Preview")
st.dataframe(X_eval.head(20), use_container_width=True)

y_true = normalize_labels(y_eval_raw, le)
y_pred = normalize_labels(model.predict(X_eval), le)
y_proba = model.predict_proba(X_eval)

cm = confusion_matrix(y_true, y_pred, labels=class_names)
metrics = compute_metrics(y_true, y_pred, y_proba)

colA, colB = st.columns(2)

with colA:
    st.markdown("### Evaluation Metrics")
    cols = st.columns(2)
    for i, (k, v) in enumerate(metrics.items()):
        cols[i % 2].metric(k, f"{v:.4f}")

with colB:
    st.markdown("### Confusion Matrix")
    st.pyplot(plot_confusion_matrix(cm, class_names), use_container_width=True)

st.markdown("### Classification Report")
st.code(classification_report(y_true, y_pred))
