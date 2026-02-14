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

    if hasattr(model, "named_steps") and "preprocess" in model.named_steps:
        pre = model.named_steps["preprocess"]

        if hasattr(pre, "feature_names_in_"):
            return list(pre.feature_names_in_)

        cols = []

        if hasattr(pre, "transformers"):
            for name, trans, colspec in pre.transformers:
                if colspec is None or colspec == "drop":
                    continue
                if colspec == "passthrough":
                    continue
                if isinstance(colspec, (list, tuple, np.ndarray, pd.Index)):
                    cols.extend(list(colspec))

        if cols:
            return _dedupe_preserve_order(cols)

        if hasattr(pre, "transformers_"):
            cols2 = []
            for name, trans, colspec in pre.transformers_:
                if colspec is None or colspec == "drop":
                    continue
                if colspec == "passthrough":
                    continue
                if isinstance(colspec, (list, tuple, np.ndarray, pd.Index)):
                    cols2.extend(list(colspec))
            if cols2:
                return _dedupe_preserve_order(cols2)

    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    return None


def align_X_to_model(X: pd.DataFrame, model):

    expected = get_expected_input_columns(model)
    if expected is None:
        return X.copy()

    X2 = X.copy()

    missing = [c for c in expected if c not in X2.columns]
    if missing:
        for c in missing:
            X2[c] = np.nan

    X2 = X2[expected]
    return X2

def normalize_to_str_labels(y, label_encoder=None):

    y_arr = np.asarray(y).ravel()
    if label_encoder is not None and np.issubdtype(y_arr.dtype, np.number):
        y_arr = label_encoder.inverse_transform(y_arr.astype(int))
    return pd.Series(y_arr).astype(str).str.strip().to_numpy()


def get_model_class_names(model, fallback_label_encoder=None):
 
    if fallback_label_encoder is not None:
        return list(fallback_label_encoder.classes_)
    try:
        return list(model.named_steps["clf"].classes_)
    except Exception:
        return None


def compute_metrics(y_true_str, y_pred_str, y_proba, class_names):
    acc = accuracy_score(y_true_str, y_pred_str)
    prec = precision_score(y_true_str, y_pred_str, average="weighted", zero_division=0)
    rec = recall_score(y_true_str, y_pred_str, average="weighted", zero_division=0)
    f1 = f1_score(y_true_str, y_pred_str, average="weighted", zero_division=0)
    mcc = matthews_corrcoef(y_true_str, y_pred_str)

    auc = np.nan
    if class_names is not None and len(class_names) >= 2:
        if len(class_names) == 2:
            auc = roc_auc_score(y_true_str, y_proba[:, 1])
        else:
            auc = roc_auc_score(
                y_true_str,
                y_proba,
                multi_class="ovr",
                average="weighted",
                labels=class_names,
            )

    return {
        "Accuracy": acc,
        "AUC": auc,
        "Precision (weighted)": prec,
        "Recall (weighted)": rec,
        "F1 (weighted)": f1,
        "MCC": mcc,
    }


def plot_confusion_matrix(cm, class_names):
    fig = plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    return fig

st.title("India Waterborne Disease Prediction (Classification Models)")
st.write(
    "This app demonstrates **multiple classification models** trained on the India Waterborne Disease dataset. "
)

global_le = load_label_encoder_if_present()

st.sidebar.header("Model Selection")
model_name = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))
model_path = os.path.join(MODEL_DIR, MODEL_FILES[model_name])

if not os.path.exists(model_path):
    st.error(f"Model file missing: {model_path}")
    st.stop()

model = load_model(model_path)
is_xgb = (model_name == "XGBoost")

if is_xgb and global_le is None:
    st.error("XGBoost selected but models/label_encoder.pkl is missing.")
    st.stop()

class_names = get_model_class_names(model, fallback_label_encoder=global_le if is_xgb else None)
if class_names is None:
    st.error("Could not infer class names from the selected model.")
    st.stop()

st.sidebar.header("Dataset Upload (CSV)")
uploaded_file = st.sidebar.file_uploader(
    "Upload test CSV",
    type=["csv"],
)
use_saved_split = st.sidebar.checkbox("If no upload, test_split.pkl will be used", value=True)

st.sidebar.markdown("---")


X_eval, y_eval_raw, source_name = None, None, None

if uploaded_file is not None:
    df_upload = pd.read_csv(uploaded_file)
    if TARGET_COL in df_upload.columns:
        y_eval_raw = df_upload[TARGET_COL]
        X_eval = df_upload.drop(columns=[TARGET_COL])
        source_name = "Uploaded CSV "
    else:
        X_eval = df_upload
        y_eval_raw = None
        source_name = "Uploaded CSV "
else:
    if use_saved_split:
        if not os.path.exists(TEST_SPLIT_PATH):
            st.error(f"Saved test split missing: {TEST_SPLIT_PATH}")
            st.stop()
        X_eval, y_eval_raw = load_test_split()
        source_name = "Saved test split "
    else:
        st.warning("Upload a CSV or enable using the saved test split.")
        st.stop()


X_eval_aligned = align_X_to_model(X_eval, model)

st.subheader("Dataset Preview")
st.write(f"**Data source:** {source_name}")
st.write(f"**Rows:** {len(X_eval_aligned)} | **Columns:** {len(X_eval_aligned.columns)}")
st.dataframe(X_eval_aligned.head(20), use_container_width=True)


st.subheader("Possible Output Classes")
st.dataframe(pd.DataFrame({"Disease Class": class_names}))

st.subheader("Evaluation (Metrics + Confusion Matrix / Classification Report)")

if y_eval_raw is None:
    st.warning(
        f"No '{TARGET_COL}' column was found, so evaluation cannot be computed. "
        f"Upload a labeled test CSV containing '{TARGET_COL}'."
    )
else:
    y_true_str = normalize_to_str_labels(y_eval_raw, label_encoder=global_le)

    y_pred_eval = model.predict(X_eval_aligned)
    y_proba_eval = model.predict_proba(X_eval_aligned)
    y_pred_str = normalize_to_str_labels(y_pred_eval, label_encoder=global_le if is_xgb else None)

    cm = confusion_matrix(y_true_str, y_pred_str, labels=class_names)
    metrics = compute_metrics(y_true_str, y_pred_str, y_proba_eval, class_names=class_names)

    colA, colB = st.columns([1, 1])
    with colA:
        st.markdown("### Metrics")
        metrics_table = pd.DataFrame([metrics]).T
        metrics_table.columns = ["Score"]
        st.table(metrics_table)

    with colB:
        st.markdown("### Confusion Matrix")
        fig = plot_confusion_matrix(cm, class_names)
        st.pyplot(fig)

    st.markdown("### Classification Report")
    report_text = classification_report(y_true_str, y_pred_str, labels=class_names, zero_division=0)
    st.code(report_text)



st.sidebar.markdown("---")

