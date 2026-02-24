"""Application configuration."""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    app_name: str = "Bank Customer Churn Prediction API"
    debug: bool = False
    log_level: str = "INFO"

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    models_dir: Path = Path(__file__).resolve().parent.parent / "models"
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"
    logs_dir: Path = Path(__file__).resolve().parent.parent / "logs"

    # Model
    default_model_filename: str = "best_model.joblib"
    default_preprocessor_filename: str = "preprocessor.joblib"
    default_metadata_filename: str = "model_metadata.json"

    # Supported algorithms for training
    supported_algorithms: list[str] = [
        "logistic_regression",
        "random_forest",
        "xgboost",
    ]

    class Config:
        env_prefix = "CHURN_"
        env_file = ".env"
        extra = "ignore"


def get_settings() -> Settings:
    """Return application settings singleton."""
    return Settings()
