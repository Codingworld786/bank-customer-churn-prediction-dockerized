"""Pydantic schemas for API request/response."""
from typing import Any, Optional
from pydantic import BaseModel, Field

#pydantic to  validate input types
# --- Single prediction input (one customer) ---
class CustomerFeatures(BaseModel):
    """Features for a single customer prediction."""

    CreditScore: int = Field(..., ge=0, le=850, description="Credit score")
    Geography: str = Field(..., min_length=1, description="Country e.g. France, Germany, Spain")
    Gender: str = Field(..., min_length=1, description="Male or Female")
    Age: int = Field(..., ge=18, le=120, description="Age in years")
    Tenure: int = Field(..., ge=0, le=10, description="Years with bank")
    Balance: float = Field(..., ge=0, description="Account balance")
    NumOfProducts: int = Field(..., ge=1, le=4, description="Number of products")
    HasCrCard: int = Field(..., ge=0, le=1, description="Has credit card (0/1)")
    IsActiveMember: int = Field(..., ge=0, le=1, description="Active member (0/1)")
    EstimatedSalary: float = Field(..., ge=0, description="Estimated salary")

    model_config = {"json_schema_extra": {"example": {
        "CreditScore": 600,
        "Geography": "France",
        "Gender": "Male",
        "Age": 40,
        "Tenure": 3,
        "Balance": 60000.0,
        "NumOfProducts": 2,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000.0,
    }}}


# --- Single prediction response ---
class SinglePredictionResponse(BaseModel):
    """Response for a single prediction."""

    churn_prediction: int = Field(..., description="0 = stay, 1 = churn")
    churn_probability: float = Field(..., ge=0, le=1, description="Probability of churn")
    customer_id: Optional[int] = Field(None, description="Optional customer index from request")


# --- Batch prediction: list of customers ---
class BatchPredictionRequest(BaseModel):
    """Request body for batch prediction."""

    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=10_000)


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction."""

    predictions: list[SinglePredictionResponse]
    total: int = Field(..., description="Number of predictions")


# --- Train request ---
class TrainRequest(BaseModel):
    """Optional parameters for training (e.g. algorithms to use)."""

    algorithms: Optional[list[str]] = Field(
        None,
        description="Algorithms to train: logistic_regression, random_forest, xgboost",
    )
    metric_for_best: Optional[str] = Field(
        "f1_score",
        description="Metric to select best model: accuracy, precision, recall, f1_score, roc_auc",
    )


# --- Train response ---
class TrainResponse(BaseModel):
    """Response after training."""

    message: str
    best_algorithm: str
    best_score: float
    metrics: dict[str, Any]
    all_results: Optional[dict[str, Any]] = None


# --- Model metadata response ---
class ModelMetadataResponse(BaseModel):
    """Model metadata and metrics."""

    loaded: bool
    best_algorithm: Optional[str] = None
    metric_used: Optional[str] = None
    best_score: Optional[float] = None
    metrics: Optional[dict[str, Any]] = None
    feature_columns: Optional[list[str]] = None
    target_column: Optional[str] = None
    message: Optional[str] = None


# --- Health response ---
class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: Optional[str] = None
    model_loaded: bool = False
