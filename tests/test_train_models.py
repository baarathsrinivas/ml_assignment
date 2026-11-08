import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from train_models import (
    build_categorical_pipeline,
    build_numeric_pipeline,
    build_preprocessor,
    coerce_numeric_columns,
    load_dataset,
    remove_unlabeled_rows,
)


def test_remove_unlabeled_rows_preserves_feature_missing_values():
    data = pd.DataFrame({"feature": [np.nan, 2.0], "quality": ["good", None]})

    cleaned, removed_count = remove_unlabeled_rows(data, "quality")

    assert removed_count == 1
    assert len(cleaned) == 1
    assert pd.isna(cleaned.iloc[0]["feature"])


def test_load_dataset_treats_question_mark_as_missing(tmp_path):
    dataset_path = tmp_path / "mushrooms.csv"
    dataset_path.write_text("class,stalk-root\ne,?\np,b\n")

    data = load_dataset(dataset_path)

    assert pd.isna(data.loc[0, "stalk-root"])


def test_coerce_numeric_columns_converts_only_fully_numeric_columns():
    features = pd.DataFrame(
        {"numeric_text": ["1.5", None, "3.5"], "mixed": ["1", "unknown", None]}
    )

    converted = coerce_numeric_columns(features)

    assert is_numeric_dtype(converted["numeric_text"])
    assert converted["numeric_text"].isna().sum() == 1
    assert converted["mixed"].dtype == object
    assert features["numeric_text"].dtype == object


def test_numeric_pipeline_imputes_missing_and_all_empty_columns():
    features = pd.DataFrame(
        {"measurement": [1.0, np.nan, 3.0], "empty": [np.nan, np.nan, np.nan]}
    )

    transformed = build_numeric_pipeline().fit_transform(features)

    assert transformed.shape == (3, 2)
    assert np.isfinite(transformed).all()


def test_categorical_pipeline_imputes_missing_and_all_empty_columns():
    features = pd.DataFrame(
        {"category": ["red", None, "blue"], "empty": [None, None, None]}
    )

    transformed = build_categorical_pipeline().fit_transform(features)

    assert transformed.shape[0] == 3
    assert np.isfinite(transformed.toarray()).all()


def test_preprocessor_handles_mixed_features_with_missing_values():
    features = pd.DataFrame(
        {
            "measurement": [1.0, np.nan, 3.0, 4.0],
            "category": ["red", None, "blue", "red"],
        }
    )

    transformed = build_preprocessor(features).fit_transform(features)

    assert transformed.shape[0] == len(features)
    assert np.isfinite(transformed).all()