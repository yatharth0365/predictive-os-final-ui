import pandas as pd
import numpy as np


class AutoDataCleaner:
    """Deterministic, stateful cleaner shared by training and inference."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df.columns = self._normalize_columns(self.df.columns)
        self.dropped_columns = []
        self.numeric_fill_values = {}
        self.categorical_fill_value = "Unknown"
        self.outlier_bounds = {}

    @staticmethod
    def _normalize_columns(columns):
        return pd.Index(columns).astype(str).str.strip().str.replace(r'[\[\]<>]', '_', regex=True)

    def fit_transform(self, threshold: float = 0.5, multiplier: float = 1.5):
        self.drop_high_missing_columns(threshold)
        self.fill_categorical_missing()
        self.fill_numerical_missing()
        self.fit_outlier_bounds(multiplier)
        self.apply_outlier_bounds()
        return self.df

    def clean_all(self):
        # Backwards-compatible training entry point.
        return self.fit_transform()

    def transform(self, df: pd.DataFrame):
        """Apply the exact training-time cleaning policy to unseen data."""
        if df.empty:
            return df
        out = df.copy()
        out.columns = self._normalize_columns(out.columns)
        out = out.drop(columns=self.dropped_columns, errors="ignore")

        for col, value in self.numeric_fill_values.items():
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(value)

        for col in out.select_dtypes(include=["object", "category"]).columns:
            out[col] = out[col].fillna(self.categorical_fill_value)

        for col, bounds in self.outlier_bounds.items():
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(self.numeric_fill_values.get(col, 0))
                out[col] = out[col].clip(bounds[0], bounds[1])
        return out

    def drop_high_missing_columns(self, threshold: float = 0.5):
        if len(self.df) == 0:
            return self.df
        missing_ratio = self.df.isnull().mean()
        columns_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
        self.dropped_columns = columns_to_drop
        self.df = self.df.drop(columns=columns_to_drop)
        return self.df

    def fill_categorical_missing(self, value: str = "Unknown"):
        self.categorical_fill_value = value
        categorical_columns = self.df.select_dtypes(include=["object", "category"]).columns
        if len(categorical_columns):
            self.df[categorical_columns] = self.df[categorical_columns].fillna(value)
        return self.df

    def fill_numerical_missing(self):
        numerical_columns = self.df.select_dtypes(include=[np.number]).columns
        self.numeric_fill_values = self.df[numerical_columns].median().to_dict()
        for col in numerical_columns:
            self.df[col] = self.df[col].fillna(self.numeric_fill_values[col])
        return self.df

    def fit_outlier_bounds(self, multiplier: float = 1.5):
        self.outlier_bounds = {}
        numerical_columns = self.df.select_dtypes(include=[np.number]).columns
        for col in numerical_columns:
            if self.df[col].nunique(dropna=True) < 10:
                continue
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            if pd.isna(iqr) or iqr == 0:
                continue
            self.outlier_bounds[col] = (q1 - multiplier * iqr, q3 + multiplier * iqr)
        return self.outlier_bounds

    def apply_outlier_bounds(self):
        for col, bounds in self.outlier_bounds.items():
            self.df[col] = self.df[col].clip(bounds[0], bounds[1])
        return self.df
