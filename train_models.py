from __future__ import annotations

from pathlib import Path
import sys
import warnings

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from imputers import DataFrameImputer

warnings.filterwarnings(
    "ignore", message="The `probability` parameter", category=FutureWarning
)


DATA_PATH = Path("data/mushrooms.csv")
TARGET_COLUMN = "class"


def load_dataset(data_path: Path) -> pd.DataFrame:
    return pd.read_csv(data_path, na_values=["?"])


def remove_unlabeled_rows(
    data: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, int]:
    unlabeled_rows = data[target].isna()
    return data.loc[~unlabeled_rows].copy(), int(unlabeled_rows.sum())


def coerce_numeric_columns(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    for column in features.columns:
        if pd.api.types.is_numeric_dtype(features[column]):
            continue
        numeric_values = pd.to_numeric(features[column], errors="coerce")
        if numeric_values.notna().sum() == features[column].notna().sum():
            features[column] = numeric_values
    return features


def build_numeric_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", DataFrameImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def build_categorical_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", DataFrameImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        column for column in features.columns if column not in numeric_columns
    ]

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", build_numeric_pipeline(), numeric_columns))
    if categorical_columns:
        transformers.append(
            ("categorical", build_categorical_pipeline(), categorical_columns)
        )

    return ColumnTransformer(transformers, sparse_threshold=0)


def calculate_auc(y_true: pd.Series, probabilities, classes) -> float:
    if len(classes) == 2:
        return roc_auc_score(y_true, probabilities[:, 1])
    return roc_auc_score(
        y_true,
        probabilities,
        labels=classes,
        multi_class="ovr",
        average="weighted",
    )


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / DATA_PATH

    data = load_dataset(data_path)
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Target column {TARGET_COLUMN!r} is not present in {data_path}.")
    data, dropped_rows = remove_unlabeled_rows(data, TARGET_COLUMN)
    if dropped_rows:
        print(
            f"Dropping {dropped_rows} row(s) without a {TARGET_COLUMN!r} label.",
            file=sys.stderr,
        )

    features = coerce_numeric_columns(data.drop(columns=TARGET_COLUMN))
    labels = data[TARGET_COLUMN]
    if len(features.columns) < 12:
        print(
            f"Warning: dataset has {len(features.columns)} feature columns; "
            "the assignment guideline recommends at least 12.",
            file=sys.stderr,
        )
    if len(data) < 500:
        raise ValueError("The assignment dataset must contain at least 500 instances.")
    if labels.nunique() < 2:
        raise ValueError("The target column must contain at least two classes.")
    if labels.value_counts().min() < 2:
        raise ValueError("Each target class needs at least two instances for a stratified split.")

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    test_data = data.loc[X_test.index]
    test_data_path = project_dir / "test_data.csv"
    test_data.to_csv(test_data_path, index=False)

    preprocessor = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_from_csv = pd.read_csv(test_data_path)
    y_test_from_csv = X_test_from_csv[TARGET_COLUMN]
    X_test_processed = preprocessor.transform(
        X_test_from_csv.drop(columns=TARGET_COLUMN)
    )

    models = {
        "Logistic Regression": (
            "logistic_regression.pkl",
            LogisticRegression(max_iter=1000),
        ),
        "Decision Tree": ("decision_tree.pkl", DecisionTreeClassifier(random_state=42)),
        "K-Nearest Neighbors": (
            "knn.pkl",
            KNeighborsClassifier(n_neighbors=min(5, len(y_train))),
        ),
        "Gaussian Naive Bayes": ("naive_bayes.pkl", GaussianNB()),
        "Support Vector Machine": ("svm.pkl", SVC(probability=True, random_state=42)),
    }

    artifact_dir = project_dir / "model"
    artifact_dir.mkdir(exist_ok=True)
    joblib.dump(preprocessor, artifact_dir / "scaler.pkl")

    metric_rows = []
    for model_name, (artifact_name, model) in models.items():
        print(f"Training {model_name}...", flush=True)
        model.fit(X_train_processed, y_train)
        predictions = model.predict(X_test_processed)
        probabilities = model.predict_proba(X_test_processed)

        joblib.dump(model, artifact_dir / artifact_name)
        metric_rows.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test_from_csv, predictions),
                "AUC": calculate_auc(y_test_from_csv, probabilities, model.classes_),
                "Precision": precision_score(
                    y_test_from_csv, predictions, average="weighted", zero_division=0
                ),
                "Recall": recall_score(y_test_from_csv, predictions, average="weighted"),
                "F1": f1_score(
                    y_test_from_csv, predictions, average="weighted", zero_division=0
                ),
                "MCC": matthews_corrcoef(y_test_from_csv, predictions),
            }
        )
        stats = metric_rows[-1]
        print(
            f"  {model_name} | "
            + " | ".join(
                f"{metric}={stats[metric]:.4f}"
                for metric in ("Accuracy", "AUC", "Precision", "Recall", "F1", "MCC")
            ),
            flush=True,
        )
        matrix = confusion_matrix(y_test_from_csv, predictions, labels=model.classes_)
        print(
            pd.DataFrame(
                matrix,
                index=[f"actual={label}" for label in model.classes_],
                columns=[f"predicted={label}" for label in model.classes_],
            ).to_string(),
            flush=True,
        )

    print(pd.DataFrame(metric_rows).to_markdown(index=False, floatfmt=".4f"))


if __name__ == "__main__":
    main()