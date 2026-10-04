import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


class FeatureBuilder:
    """Fits preprocessing once and guarantees a stable feature schema at inference."""

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.categorical_cols = []
        self.numerical_cols = []
        self.feature_columns = []

    def extract_datetime_features(self, df: pd.DataFrame):
        df = df.copy()
        for col in list(df.columns):
            if df[col].dtype == "object":
                parsed = pd.to_datetime(df[col], errors="coerce")
                # Only convert when most non-null values are actually dates.
                non_null = df[col].notna().sum()
                if non_null and parsed.notna().sum() / non_null >= 0.8:
                    df[col] = parsed

        datetime_columns = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
        for col in datetime_columns:
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day
            df[f"{col}_dayofweek"] = df[col].dt.dayofweek
        return df.drop(columns=datetime_columns, errors="ignore")

    def fit_transform(self, df: pd.DataFrame, target_col: str = None):
        df = self.extract_datetime_features(df)
        feature_cols = [c for c in df.columns if c != target_col]
        self.feature_columns = feature_cols.copy()
        self.categorical_cols = df[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()
        self.numerical_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

        for col in self.categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        if self.numerical_cols:
            df[self.numerical_cols] = self.scaler.fit_transform(df[self.numerical_cols])

        return df[self.feature_columns]

    def transform(self, df: pd.DataFrame, target_col: str = None):
        df = self.extract_datetime_features(df)

        # Build exactly the training schema. Ignore unexpected inference columns.
        for col in self.feature_columns:
            if col not in df.columns:
                if col in self.categorical_cols:
                    df[col] = "Unknown"
                else:
                    df[col] = 0

        for col in self.categorical_cols:
            le = self.label_encoders[col]
            known_classes = set(le.classes_)
            df[col] = df[col].astype(str).map(lambda s: le.transform([s])[0] if s in known_classes else len(le.classes_))

        if self.numerical_cols:
            for col in self.numerical_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            df[self.numerical_cols] = self.scaler.transform(df[self.numerical_cols])

        return df[self.feature_columns]
