"""Custom exceptions for the churn prediction service."""


class ChurnPredictionError(Exception):
    """Base exception for churn prediction."""

    pass


class ModelNotTrainedError(ChurnPredictionError):
    """Raised when prediction is attempted but no model is loaded."""

    pass


class InvalidInputError(ChurnPredictionError):
    """Raised when input data is invalid."""

    pass


class TrainingError(ChurnPredictionError):
    """Raised when model training fails."""

    pass


class PreprocessingError(ChurnPredictionError):
    """Raised when data preprocessing fails."""

    pass
