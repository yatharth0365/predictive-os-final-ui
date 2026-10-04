import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.preprocessing import LabelEncoder


class ModelEngine:
    def __init__(self, task_type: str = "classification"):
        self.task_type = task_type
        self.best_model = None
        self.best_model_name = ""
        self.models = {}
        self.target_encoder = None
        self.cv_folds = 5

    def _initialize_models(self):
        if self.task_type == "classification":
            self.models = {
                "RandomForest": RandomForestClassifier(n_estimators=250, random_state=42, n_jobs=-1, class_weight="balanced"),
                "XGBoost": XGBClassifier(n_estimators=250, random_state=42, eval_metric="logloss", n_jobs=-1),
                "LightGBM": LGBMClassifier(n_estimators=250, random_state=42, verbosity=-1, n_jobs=-1),
            }
        else:
            self.models = {
                "RandomForest": RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1),
                "XGBoost": XGBRegressor(n_estimators=250, random_state=42, n_jobs=-1),
                "LightGBM": LGBMRegressor(n_estimators=250, random_state=42, verbosity=-1, n_jobs=-1),
            }

    def train_and_benchmark(self, X, y):
        self._initialize_models()
        y = y.copy()

        if self.task_type == "classification":
            if y.dtype == "object" or y.dtype.name == "category":
                self.target_encoder = LabelEncoder()
                y = self.target_encoder.fit_transform(y.astype(str))
            else:
                y = np.asarray(y)

            counts = np.bincount(y.astype(int))
            min_class_count = counts[counts > 0].min() if len(counts) else 0
            self.cv_folds = min(5, int(min_class_count))
            if self.cv_folds < 2:
                raise ValueError("Classification requires at least 2 samples in every class for cross-validation.")
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            scoring = "f1_macro"
        else:
            self.cv_folds = min(5, len(X))
            if self.cv_folds < 2:
                raise ValueError("At least 2 rows are required for regression.")
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            scoring = "r2"

        results = {}
        for name, model in self.models.items():
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
            results[name] = float(np.mean(scores))

        best_model_name = max(results, key=results.get)
        self.best_model_name = best_model_name
        self.best_model = self.models[best_model_name]
        self.best_model.fit(X, y)
        return results, best_model_name
