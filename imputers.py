import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameImputer(BaseEstimator, TransformerMixin):
    def __init__(self, strategy: str):
        self.strategy = strategy

    def fit(self, features: pd.DataFrame, y=None):
        if self.strategy not in {"median", "most_frequent"}:
            raise ValueError(f"Unsupported imputation strategy: {self.strategy!r}.")

        self.feature_names_in_ = list(features.columns)
        self.fill_values_ = {}
        for column in self.feature_names_in_:
            if self.strategy == "median":
                value = features[column].median()
                self.fill_values_[column] = 0 if pd.isna(value) else value
            else:
                modes = features[column].mode(dropna=True)
                self.fill_values_[column] = (
                    modes.iloc[0] if not modes.empty else "__missing__"
                )
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "fill_values_"):
            raise RuntimeError("DataFrameImputer must be fitted before transform.")

        missing_columns = [
            column for column in self.feature_names_in_ if column not in features.columns
        ]
        if missing_columns:
            raise ValueError(f"Missing columns required for imputation: {missing_columns}.")

        return features.loc[:, self.feature_names_in_].fillna(self.fill_values_)