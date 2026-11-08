from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from train_models import TARGET_COLUMN, calculate_auc, coerce_numeric_columns, load_dataset


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"
TEST_DATA_PATH = PROJECT_DIR / "test_data.csv"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Gaussian Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "svm.pkl",
}
METRIC_NAMES = ("Accuracy", "AUC", "Precision", "Recall", "F1", "MCC")


def load_test_data(uploaded_file) -> tuple[pd.DataFrame, str]:
    if uploaded_file is not None:
        return load_dataset(uploaded_file), uploaded_file.name
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            "Upload a labeled test CSV or create test_data.csv by running train_models.py."
        )
    return load_dataset(TEST_DATA_PATH), TEST_DATA_PATH.name


def evaluate_model(
    data: pd.DataFrame, model, preprocessor
) -> tuple[dict[str, float], pd.Series, pd.Series]:
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"The CSV must include the true label column '{TARGET_COLUMN}'.")
    if data[TARGET_COLUMN].isna().any():
        raise ValueError(f"The '{TARGET_COLUMN}' column contains missing labels.")

    features = coerce_numeric_columns(data.drop(columns=TARGET_COLUMN))
    expected_features = list(preprocessor.feature_names_in_)
    missing_features = [name for name in expected_features if name not in features.columns]
    if missing_features:
        raise ValueError(f"Missing required feature columns: {', '.join(missing_features)}")
    features = features.loc[:, expected_features]

    actual = data[TARGET_COLUMN]
    processed_features = preprocessor.transform(features)
    predicted = pd.Series(model.predict(processed_features), index=actual.index)
    probabilities = model.predict_proba(processed_features)
    try:
        auc = calculate_auc(actual, probabilities, model.classes_)
    except ValueError:
        auc = float("nan")

    metrics = {
        "Accuracy": accuracy_score(actual, predicted),
        "AUC": auc,
        "Precision": precision_score(actual, predicted, average="weighted", zero_division=0),
        "Recall": recall_score(actual, predicted, average="weighted", zero_division=0),
        "F1": f1_score(actual, predicted, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(actual, predicted),
    }
    return metrics, actual, predicted


st.set_page_config(page_title="Mushroom Classification", page_icon="M", layout="wide")
st.markdown(
    """
    <style>
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Mushroom Edibility Classifier")
st.write("Predict the dataset's edible or poisonous class from mushroom characteristics.")

problem_column, dataset_column = st.columns([1, 1.3])
with problem_column:
    st.subheader("Problem Statement")
    st.write(
        "Compare five classifiers on identifying the recorded edibility class of "
        "gilled mushrooms. The target is `class`: `e` means edible and `p` means "
        "poisonous or of unknown edibility."
    )
    st.warning(
        "For education and model evaluation only. Never use a model prediction to "
        "decide whether a wild mushroom is safe to eat."
    )

with dataset_column:
    st.subheader("Dataset")
    st.write(
        "The UCI Mushroom dataset describes hypothetical specimens from 23 species "
        "of gilled mushrooms. The unknown-edibility class is combined with poisonous."
    )
    dataset_metrics = st.columns(3)
    dataset_metrics[0].metric("Instances", "8,124")
    dataset_metrics[1].metric("Attributes", "22 + class")
    dataset_metrics[2].metric("Classes", "2")
    st.caption("Class balance: 4,208 edible · 3,916 poisonous/unknown")

with st.expander("Dataset attributes"):
    st.write(
        "Cap shape, surface, and color; bruising; odor; gill attachment, spacing, "
        "size, and color; stalk shape, root, surfaces, and colors; veil type and "
        "color; ring number and type; spore-print color; population; and habitat."
    )

with st.sidebar:
    st.header("Evaluation")
    uploaded_file = st.file_uploader("Test data (CSV)", type=["csv"])
    selected_model = st.selectbox("Classifier", list(MODEL_FILES))
    if TEST_DATA_PATH.exists():
        st.download_button(
            "Download default test data",
            data=TEST_DATA_PATH.read_bytes(),
            file_name=TEST_DATA_PATH.name,
            mime="text/csv",
        )

try:
    test_data, data_source = load_test_data(uploaded_file)
    model_path = MODEL_DIR / MODEL_FILES[selected_model]
    preprocessor_path = MODEL_DIR / "scaler.pkl"
    if not model_path.exists() or not preprocessor_path.exists():
        raise FileNotFoundError("Trained model files are missing. Run train_models.py first.")
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    metrics, actual, predicted = evaluate_model(test_data, model, preprocessor)
except Exception as error:
    st.error(str(error))
    st.stop()

st.caption(f"{data_source} · {len(test_data):,} labeled records · {selected_model}")
metric_columns = st.columns(6)
for column, metric_name in zip(metric_columns, METRIC_NAMES):
    value = metrics[metric_name]
    display_value = f"{value:.4f}" if pd.notna(value) else "N/A"
    column.metric(metric_name, display_value)

if pd.isna(metrics["AUC"]):
    st.info("AUC is unavailable because the test data does not contain every target class.")

matrix_column, report_column = st.columns(2)
with matrix_column:
    st.subheader("Confusion Matrix")
    figure, axis = plt.subplots(figsize=(6, 4))
    ConfusionMatrixDisplay.from_predictions(
        actual,
        predicted,
        labels=model.classes_,
        display_labels=model.classes_,
        cmap="Blues",
        colorbar=False,
        ax=axis,
    )
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)

with report_column:
    st.subheader("Classification Report")
    report = classification_report(
        actual,
        predicted,
        labels=model.classes_,
        output_dict=True,
        zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).transpose().round(3), width="stretch")