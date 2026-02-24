"""Churn prediction service: load model, predict, train."""
import json
from pathlib import Path
from typing import Any, Optional

import joblib #joblib = save trained model to file and load it later without retraining.
import pandas as pd

from app.config import get_settings
from app.core.preprocessing import ChurnPreprocessor, FEATURE_COLUMNS, TARGET_COLUMN
from app.core.models import ModelTrainer, ALGORITHMS
from app.utils.logging import get_logger
from app.utils.exceptions import ModelNotTrainedError, TrainingError, PreprocessingError

logger = get_logger(__name__)


class ChurnService:
    """Singleton-style service for model and preprocessor lifecycle."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._model: Any = None
        self._preprocessor: Optional[ChurnPreprocessor] = None
        self._metadata: dict[str, Any] = {}
        self._load_model_if_exists()

    def _model_path(self) -> Path:
        return self._settings.models_dir / self._settings.default_model_filename #models/best_model.joblib

    def _preprocessor_path(self) -> Path:
        return self._settings.models_dir / self._settings.default_preprocessor_filename

    def _metadata_path(self) -> Path:
        return self._settings.models_dir / self._settings.default_metadata_filename

    def _load_model_if_exists(self) -> None:
        """Load saved model and preprocessor if files exist."""
        if self._model_path().exists() and self._preprocessor_path().exists():
            try:
                self._model = joblib.load(self._model_path())
                self._preprocessor = ChurnPreprocessor.load(self._preprocessor_path())
                if self._metadata_path().exists():
                    with open(self._metadata_path()) as f:
                        self._metadata = json.load(f)
                logger.info("Model and preprocessor loaded from disk.")
            except Exception as e:
                logger.exception("Failed to load model: %s", e)
                self._model = None
                self._preprocessor = None
                self._metadata = {}

    def is_loaded(self) -> bool:
        return self._model is not None and self._preprocessor is not None

    def predict_single(self, features: dict[str, Any]) -> tuple[int, float]:
        """Predict churn for one customer. Returns (prediction 0/1, probability)."""
        if not self.is_loaded():
            raise ModelNotTrainedError("No model loaded. Train a model first via POST /train.")
        df = pd.DataFrame([features])
        X = self._preprocessor.transform(df)
        pred = int(self._model.predict(X)[0])
        proba = float(self._model.predict_proba(X)[0, 1])
        return pred, proba

    def predict_batch(self, customers: list[dict[str, Any]]) -> list[tuple[int, float]]:
        """Predict churn for multiple customers."""
        if not self.is_loaded():
            raise ModelNotTrainedError("No model loaded. Train a model first via POST /train.")
        df = pd.DataFrame(customers)
        X = self._preprocessor.transform(df)
        preds = self._model.predict(X)
        probas = self._model.predict_proba(X)[:, 1]
        return [(int(p), float(pr)) for p, pr in zip(preds, probas)]

    def train(self, csv_path: Path, algorithms: Optional[list[str]] = None) -> dict[str, Any]:
        """Train models from CSV and save best. CSV must have Exited column."""
        if not csv_path.exists():
            raise TrainingError(f"Data file not found: {csv_path}")
        df = pd.read_csv(csv_path)
        preprocessor = ChurnPreprocessor()
        preprocessor.fit(df)
        trainer = ModelTrainer(
            models_dir=self._settings.models_dir,
            preprocessor=preprocessor,
            metric_for_best="f1_score",
        )
        result = trainer.train(df, algorithms=algorithms)
        self._load_model_if_exists()
        return result

    def get_metadata(self) -> dict[str, Any]:
        """Return current metadata (from file or in-memory)."""
        if self._metadata_path().exists():
            try:
                with open(self._metadata_path()) as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "loaded": self.is_loaded(),
            "best_algorithm": self._metadata.get("best_algorithm"),
            "metric_used": self._metadata.get("metric_used"),
            "best_score": self._metadata.get("best_score"),
            "metrics": self._metadata.get("metrics"),
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
        }


# Module-level singleton for FastAPI dependency
_churn_service: Optional[ChurnService] = None


def get_churn_service() -> ChurnService:
    global _churn_service
    if _churn_service is None:
        _churn_service = ChurnService()
    return _churn_service
