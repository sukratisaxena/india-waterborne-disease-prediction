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
    "Naive Bayes (Multinomial)": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
}

TEST_SPLIT_PATH = os.path.join(MODEL_DIR, "test_split.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")

@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)

@st.cache_resource
def load_test_split():
    return joblib.load(TEST_SPLIT_PATH)

@st.cache_resource
def load_label_encoder_if_present():
    if os.path.exists(LABEL_ENCODER_PATH):
        return joblib.load(LABEL_ENCODER_PATH)
    return None

def _dedupe_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def get_expected_input_columns(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return None

def align_X_to_model(X: pd.DataFrame, model):
    expected = get_expected_input_columns(model)
    if expected is None:
        return X.copy()
    X2 = X.copy()
    for c in expected:
        if c not in X2.columns:
            X2[c] = np.nan
    return X2[expected]

def normalize_to_str_labels(y, label_encoder=None):
    y_arr = np.asarray(y).ravel()
    if label_encoder is not None and np.issubdtype(y_arr.dtype, np.number):
        y_arr = label_encoder.inverse_transform(y_arr.astype(int))
    return pd.Series(y_arr).astype(str).to_numpy()

def get_model_class_names(model, fallback_label_encoder=None):
    if fallback_label_encoder is not None:
        return list(fallback_label_encoder.classes_)
    try:
        return list(model.classes_)
    except:
        return None

def compute_metrics(y_true_str, y_pred_str, y_proba, class_names):
    return {
        "Accuracy": accuracy_score(y_true_str, y_pred_str),
        "AUC": roc_auc_score(y_true_str, y_proba, multi_class="ovr", average="weighted"),
        "Precision": precision_score(y_true_str, y_pred_str, average="weighted", zero_division=0),
        "Recall": recall_score(y_true_str, y_pred_str, average="weighted", zero_division=0),
        "F1": f1_score(y_true_str, y_pred_str, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true_str, y_pred_str),
    }

def plot_confusion_matrix(cm, class_names):
    fig = plt.figure(figsize=(15, 14))

    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix", fontsize=26)
    plt.xlabel("Predicted", fontsize=20)
    plt.ylabel("Actual", fontsize=20)

    plt.xticks(
        range(len(class_names)),
        class_names,
        rotation=45,
        ha="right",
        fontsize=16
    )
    plt.yticks(
        range(len(class_names)),
        class_names,
        fontsize=16
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, cm[i, j],
                ha="center", va="center",
                fontsize=16,
                color="white" if cm[i, j] > cm.max()/2 else "black"
            )

    plt.colorbar()
    plt.tight_layout()
    return fig

st.title("India Waterborne Disease Prediction")

global_le = load_label_encoder_if_present()

st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))
model_path = os.path.join(MODEL_DIR, MODEL_FILES[model_name])
model = load_model(model_path)


class_names = get_model_class_names(model, fallback_label_encoder=global_le)

st.sidebar.header("Dataset Upload")
uploaded_file = st.sidebar.file_uploader("Upload test CSV", type=["csv"])
use_saved_split = st.sidebar.checkbox("Use saved test split", value=True)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    y_eval_raw = df[TARGET_COL]
    X_eval = df.drop(columns=[TARGET_COL])
    source_name = "Uploaded CSV"
else:
    X_eval, y_eval_raw = load_test_split()
    source_name = "Saved test split"

X_eval = align_X_to_model(X_eval, model)

col1, col2, col3 = st.columns(3)
col1.metric("Model", model_name)
col2.metric("Samples", len(X_eval))
col3.metric("Features", len(X_eval.columns))

st.subheader("Dataset Preview")
st.write(f"Source: {source_name}")
st.dataframe(X_eval.head(20), use_container_width=True)

st.subheader(f"Evaluation — {model_name}")

y_true_str = normalize_to_str_labels(y_eval_raw, label_encoder=global_le)
y_pred = model.predict(X_eval)
y_proba = model.predict_proba(X_eval)
y_pred_str = normalize_to_str_labels(y_pred, label_encoder=global_le)

cm = confusion_matrix(y_true_str, y_pred_str, labels=class_names)
metrics = compute_metrics(y_true_str, y_pred_str, y_proba, class_names)

colA, colB = st.columns([1, 1])

with colA:
    st.markdown("### Metrics")
    for k, v in metrics.items():
        st.metric(k, f"{v:.4f}")

with colB:
    st.markdown("### Confusion Matrix")
    st.pyplot(plot_confusion_matrix(cm, class_names), use_container_width=True)


st.markdown("### Classification Report")
st.code(classification_report(y_true_str, y_pred_str))
