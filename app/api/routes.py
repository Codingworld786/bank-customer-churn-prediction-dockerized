"""API route handlers."""
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form

from app.api.schemas import (
    CustomerFeatures,
    SinglePredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    TrainResponse,
    ModelMetadataResponse,
    HealthResponse,
)
from app.services.churn_service import ChurnService, get_churn_service

from app.utils.exceptions import (
    ModelNotTrainedError,
    InvalidInputError,
    TrainingError,
    PreprocessingError,
)
from app.utils.logging import get_logger
from app import __version__
from app.config import get_settings

logger = get_logger(__name__)
router = APIRouter() #APIRouter is a class that helps to define the routes for the API.
settings = get_settings()

#health check endpoint for readiness/liveness probes. used to check if the server is running and healthy.
@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Health check endpoint for readiness/liveness probes."""
    svc = get_churn_service()#singleton instance of ChurnService.
    return HealthResponse(
        status="healthy",
        version=__version__,
        model_loaded=svc.is_loaded(), #returns True if model is loaded, False otherwise.
    )


@router.get("/metadata", response_model=ModelMetadataResponse)
def get_metadata(svc: ChurnService = Depends(get_churn_service)) -> ModelMetadataResponse: #“Before running this endpoint, give me the ChurnService instance.”
    """Return metadata of the currently loaded model (algorithm, metrics, features)."""
    meta = svc.get_metadata() #returns metadata of the currently loaded model.
    return ModelMetadataResponse(
        loaded=svc.is_loaded(), #returns True if model is loaded, False otherwise.
        best_algorithm=meta.get("best_algorithm"), #returns the best algorithm used for training.
        metric_used=meta.get("metric_used"), #returns the metric used for training.
        best_score=meta.get("best_score"), #returns the best score achieved.
        metrics=meta.get("metrics"), #returns the metrics used for training.
        feature_columns=meta.get("feature_columns"), #returns the feature columns used for training.
        target_column=meta.get("target_column"), #returns the target column used for training.
        message=None if svc.is_loaded() else "No model loaded. Train via POST /train.", #returns a message if the model is not loaded.
    )


@router.post("/predict", response_model=SinglePredictionResponse)
def predict_single(
    body: CustomerFeatures,
    customer_id: Optional[int] = None,
    svc: ChurnService = Depends(get_churn_service),
) -> SinglePredictionResponse:
    """Predict churn for a single customer."""
    try:
        pred, proba = svc.predict_single(body.model_dump())
        print("proba", proba)
        return SinglePredictionResponse(
            churn_prediction=pred,
            churn_probability=round(proba, 6),#rounding the probability to 6 decimal places.
            customer_id=customer_id,
        )
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except (InvalidInputError, PreprocessingError) as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(
    body: BatchPredictionRequest,
    svc: ChurnService = Depends(get_churn_service),
) -> BatchPredictionResponse:
    """Predict churn for multiple customers."""
    try:
        customers = [c.model_dump() for c in body.customers]
        results = svc.predict_batch(customers)
        predictions = [
            SinglePredictionResponse(
                churn_prediction=p,
                churn_probability=round(prob, 6),
                customer_id=i,
            )
            for i, (p, prob) in enumerate(results)
        ]
        return BatchPredictionResponse(predictions=predictions, total=len(predictions))
    except ModelNotTrainedError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except (InvalidInputError, PreprocessingError) as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/train", response_model=TrainResponse)
async def train_model(
    svc: ChurnService = Depends(get_churn_service),
    file: Optional[UploadFile] = File(None),
    algorithms: Optional[str] = Form(None),
    metric_for_best: Optional[str] = Form("f1_score"),
) -> TrainResponse:
    """
    Train models from CSV. Either upload a CSV file or use default dataset (if present).
    CSV must include columns: CreditScore, Geography, Gender, Age, Tenure, Balance,
    NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Exited.
    """
    csv_path: Optional[Path] = None
    algo_list: Optional[list[str]] = None
    if algorithms:
        algo_list = [a.strip() for a in algorithms.split(",") if a.strip()]

    if file and file.filename:
        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="File must be a CSV.")
        try:
            content = await file.read()
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as tmp:
                tmp.write(content)
                csv_path = Path(tmp.name)
        except Exception as e:
            logger.exception("Failed to save uploaded file")
            raise HTTPException(status_code=500, detail=f"Upload failed: {e}")
    else:
        default_data = settings.data_dir / "Churn_Modelling.csv"
        if default_data.exists():
            csv_path = default_data
        else:
            raise HTTPException(
                status_code=400,
                detail="No file uploaded and no default data at data/Churn_Modelling.csv. Upload a CSV.",
            )

    try:
        result = svc.train(csv_path, algorithms=algo_list)
        return TrainResponse(
            message="Training completed. Best model saved.",
            best_algorithm=result["best_algorithm"],
            best_score=result["best_score"],
            metrics=result["metrics"],
            all_results=result.get("all_results"),
        )
    except TrainingError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if csv_path and file and file.filename and str(csv_path).startswith(tempfile.gettempdir()):
            try:
                csv_path.unlink(missing_ok=True)
            except Exception:
                pass
