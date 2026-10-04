import pandas as pd
import numpy as np


class PipelinePredictor:
    def __init__(self, cleaner, feature_builder, model_engine):
        self.cleaner = cleaner
        self.feature_builder = feature_builder
        self.model_engine = model_engine

    def predict_unseen(self, raw_unseen_df, confidence_threshold=75.0):
        raw = raw_unseen_df.copy()
        cleaned_df = self.cleaner.transform(raw)
        X = self.feature_builder.transform(cleaned_df)
        predictions = self.model_engine.best_model.predict(X)
        result = raw.copy()

        if self.model_engine.target_encoder is not None:
            predictions = self.model_engine.target_encoder.inverse_transform(np.asarray(predictions).astype(int))
        result["Prediction"] = predictions

        if self.model_engine.task_type == "classification" and hasattr(self.model_engine.best_model, "predict_proba"):
            probabilities = self.model_engine.best_model.predict_proba(X)
            confidence = np.max(probabilities, axis=1) * 100
            result["Confidence Score (%)"] = np.round(confidence, 2)
            result["Decision Status"] = np.where(
                confidence < confidence_threshold,
                "Human Review",
                "Auto Decision"
            )
            result["Requires Human Audit"] = confidence < confidence_threshold

        return result
