"""Data preprocessing for churn prediction."""
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
from typing import Optional

from app.utils.logging import get_logger
from app.utils.exceptions import PreprocessingError

logger = get_logger(__name__)

# Expected columns for churn dataset
FEATURE_COLUMNS = [
    "CreditScore",
    "Geography",
    "Gender",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]
TARGET_COLUMN = "Exited"
NUMERIC_COLUMNS = [
    "CreditScore",
    "Age",
    "Tenure",
    "Balance",
    "NumOfProducts",
    "HasCrCard",
    "IsActiveMember",
    "EstimatedSalary",
]
CATEGORICAL_COLUMNS = ["Geography", "Gender"]


class ChurnPreprocessor:
    """Preprocessor that encodes categoricals and scales numerics."""

    def __init__(self) -> None:
        self._fitted: bool = False
        self._column_transformer: Optional[ColumnTransformer] = None
        self._feature_names_out: Optional[list[str]] = None

    def _build_transformer(self) -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline([("scaler", StandardScaler())]), #bring mean to 0 and std dev to 1
                    NUMERIC_COLUMNS,
                ),
                (
                    "cat",
                    OneHotEncoder(
                        drop="first", #to avoid mulitcollinearity
                        sparse_output=False,
                        handle_unknown="ignore",
                    ),
                    CATEGORICAL_COLUMNS,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    def fit(self, df: pd.DataFrame) -> "ChurnPreprocessor":
        """Fit preprocessor on training data."""
        missing = set(FEATURE_COLUMNS) - set(df.columns)
        if missing:
            raise PreprocessingError(f"Missing feature columns: {missing}")

        X = df[FEATURE_COLUMNS].copy()
        X[CATEGORICAL_COLUMNS] = X[CATEGORICAL_COLUMNS].astype(str) # forcefully convert cat cols to string
        self._column_transformer = self._build_transformer()
        self._column_transformer.fit(X) # during fit num cols becomes standardise (mean 0 and variance 1)and cat cols becomes onehot encoded 
        self._feature_names_out = self._get_feature_names()
        self._fitted = True
        logger.info("Preprocessor fitted. Output features: %s", len(self._feature_names_out))
        return self

    def _get_feature_names(self) -> list[str]:
        if self._column_transformer is None:
            return []
        if hasattr(self._column_transformer, "get_feature_names_out"):
            return self._column_transformer.get_feature_names_out().tolist()
        return []

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted preprocessor."""
        if not self._fitted or self._column_transformer is None:
            raise PreprocessingError("Preprocessor not fitted. Call fit() first.")

        X = df[FEATURE_COLUMNS].copy()
        X[CATEGORICAL_COLUMNS] = X[CATEGORICAL_COLUMNS].astype(str)
        return self._column_transformer.transform(X)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)

    @property
    def feature_names_out(self) -> list[str]:
        if self._feature_names_out is None:
            return []
        return self._feature_names_out

    def save(self, path: Path) -> None:
        """Persist preprocessor to disk."""
        if not self._fitted:
            raise PreprocessingError("Cannot save unfitted preprocessor.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "transformer": self._column_transformer,
                "feature_names_out": self._feature_names_out,
                "fitted": self._fitted,
            },
            path,
        )
        logger.info("Preprocessor saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "ChurnPreprocessor":
        """Load preprocessor from disk."""
        obj = joblib.load(path)
        p = cls()
        p._column_transformer = obj["transformer"]
        p._feature_names_out = obj.get("feature_names_out", [])
        p._fitted = obj.get("fitted", True)
        return p

#####summary
        """
Load saved preprocessor

Call .transform()

It uses saved:

means

stds

category mapping

Generates numeric vector

Pass vector into model

Model outputs probability
        """