"""FastAPI application entrypoint."""
from contextlib import asynccontextmanager
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.api.routes import router
from app.utils.logging import setup_logging, get_logger
from app.utils.exceptions import ChurnPredictionError
from app.services.churn_service import get_churn_service

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: ensure dirs and load model. Shutdown: nothing special."""
    settings = get_settings()
    setup_logging(
        log_level=settings.log_level,
        log_dir=settings.logs_dir,
        log_file="churn_api.log",
    )
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    get_churn_service()
    yield
    # shutdown
    pass


app = FastAPI(
    title=get_settings().app_name,
    version="1.0.0",
    description="Predict bank customer churn from credit score, geography, gender, age, tenure, balance, products, card, activity, salary.",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["churn"])


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log each incoming HTTP request and its response time/status."""
    start = time.perf_counter()
    logger.info("HTTP %s %s", request.method, request.url.path)
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled error during %s %s", request.method, request.url.path)
        raise
    process_time_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "HTTP %s %s completed in %.2f ms with status %s",
        request.method,
        request.url.path,
        process_time_ms,
        response.status_code,
    )
    return response


@app.exception_handler(ChurnPredictionError)
async def churn_exception_handler(_request: Request, exc: ChurnPredictionError):
    """Handle custom churn errors with 500 and message."""
    logger.error("ChurnPredictionError: %s", exc)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/")
def root():
    return {
        "service": "Bank Customer Churn Prediction API",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
