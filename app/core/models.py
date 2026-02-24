"""Model training with multiple algorithms and best-model selection."""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import xgboost as xgb

from app.core.preprocessing import ChurnPreprocessor, FEATURE_COLUMNS, TARGET_COLUMN
from app.core.evaluation import evaluate_model
from app.utils.logging import get_logger
from app.utils.exceptions import TrainingError, PreprocessingError

logger = get_logger(__name__)

ALGORITHMS = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    ),
    "xgboost": xgb.XGBClassifier(
        n_estimators=100,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
        scale_pos_weight=1,
    ),
}


class ModelTrainer:
    """Train multiple models, evaluate, and persist the best one."""

    def __init__(
        self,
        models_dir: Path,
        preprocessor: ChurnPreprocessor,
        metric_for_best: str = "f1_score",
        cv_folds: int = 5,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.preprocessor = preprocessor
        self.metric_for_best = metric_for_best
        self.cv_folds = cv_folds
        self.best_algorithm: Optional[str] = None
        self.best_score: Optional[float] = None
        self.best_model: Any = None
        self.best_metrics: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}

    def _get_proba(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Get probability for positive class (churn=1)."""
        proba = model.predict_proba(X) # using inbuit func for probablity cal
        # Logistic Regression → sigmoid function internally, Random Forest → 0-1, XGBoost → 0-1
        # proba.shape[1] == 2: means binary classification
        # proba[:, 1] → probability of positive class (churn=1)
        # proba[:, 0] → probability of negative class (churn=0)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba[:, 0]

    def train(
        self,
        df: pd.DataFrame,
        algorithms: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Train specified algorithms (or all), evaluate, select best by metric,
        and save model, preprocessor, and metadata.
        """
        print("#"*50)
        print("df shape", df.head())
        print("#"*50)
        if TARGET_COLUMN not in df.columns:
            raise TrainingError(f"Target column '{TARGET_COLUMN}' not in data.")

        X = self.preprocessor.transform(df) #X → numeric matrix (scaled + encoded)
        y = df[TARGET_COLUMN].values #y → array like [1, 0, 1]

        algorithms = algorithms or list(ALGORITHMS.keys())
        results = {}
        best_score = -np.inf
        best_name = None
        best_estimator = None
        best_metrics = {}

        for name in algorithms:
            if name not in ALGORITHMS:
                logger.warning("Unknown algorithm %s, skipping.", name)
                continue
            logger.info("Training %s...", name)
            model = ALGORITHMS[name]
            model = model.fit(X, y) #here wahtever model is passed it will be fitted on the data
            y_pred = model.predict(X)
            y_prob = self._get_proba(model, X)
            metrics = evaluate_model(y, y_pred, y_prob)
            results[name] = metrics

            score = metrics.get(self.metric_for_best)
            if score is None:
                score = metrics.get("accuracy", 0)
            if score > best_score:
                best_score = score
                best_name = name
                best_estimator = model
                best_metrics = metrics

        if best_name is None or best_estimator is None:
            raise TrainingError("No model was trained successfully.")

        self.best_algorithm = best_name
        self.best_score = best_score
        self.best_model = best_estimator
        self.best_metrics = best_metrics

        self.models_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.models_dir / "best_model.joblib"
        preprocessor_path = self.models_dir / "preprocessor.joblib"
        metadata_path = self.models_dir / "model_metadata.json"

        joblib.dump(best_estimator, model_path)
        self.preprocessor.save(preprocessor_path)

        self.metadata = {
            "best_algorithm": best_name,
            "metric_used": self.metric_for_best,
            "best_score": float(best_score),
            "metrics": best_metrics,
            "all_results": {k: {kk: vv for kk, vv in v.items() if kk != "classification_report"} for k, v in results.items()},
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
        }
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        logger.info("Best model: %s (%s=%.4f)", best_name, self.metric_for_best, best_score)
        return {
            "best_algorithm": best_name,
            "best_score": float(best_score),
            "metrics": best_metrics,
            "all_results": results,
        }
############## flow of the code ##############
"""
Training Phase:
Raw DataFrame
   ↓
Preprocessor.transform()
   ↓
Numeric Matrix X
   ↓
Train Logistic / RF / XGB
   ↓
Evaluate metrics
   ↓
Select best model
   ↓
Save model + preprocessor + metadata

# prediction phase
New JSON input
   ↓
Load preprocessor
   ↓
transform()
   ↓
Load best_model
   ↓
predict_proba()
   ↓
Return churn_probability

"""