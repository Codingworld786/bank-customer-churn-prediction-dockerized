## File-wise Project Overview

This document walks through each important file in the project, what it contains, how it fits into the overall flow, and the design thinking behind it.

---

### Root level

- **`README.md`**  
  High-level project overview for users: features, project structure, how to run locally and with Docker, main endpoints, and CSV format. It is aimed at quick onboarding rather than deep architecture.

- **`PROJECT_DETAILS.md`**  
  Deep-dive documentation: full plan-to-execution story, models and rationale, tech stack choices, advantages, scope for improvement, how to add new models, and how to extend the system with LLMs. This is the strategic/architectural design doc.

- **`PROJECT_FILE_DESCRIPTIONS.md`** (this file)  
  Per-file explanation of responsibilities and how each piece connects in the runtime flow.

- **`Dockerfile`**  
  Builds a minimal Python 3.11 container image, installs `curl` (for Docker healthcheck), installs dependencies from `requirements.txt`, copies `app/` and `data/`, creates `models/` and `logs/`, and runs the API with `uvicorn`. Design intent: small, reproducible image optimized for deployment.

- **`docker-compose.yml`**  
  Orchestrates a single `churn-api` service: builds the image, exposes port `8000`, wires environment variables, mounts host volumes for `models/`, `logs/`, and `data/`, and defines a healthcheck that calls `GET /api/v1/health`. Design intent: simple, dev-friendly way to run the full stack with persistence.

- **`.dockerignore`**  
  Excludes caches, virtualenvs, logs, git files, etc. from build context to keep Docker images small and builds fast.

- **`data/Churn_Modelling.csv`**  
  Sample training dataset with the expected schema (including `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`, `Exited`). Design intent: allow immediate end-to-end testing without needing an external dataset.

- **`scripts/generate_sample_data.py`**  
  Script to synthesize a larger, realistic churn dataset with the same feature schema and save it to `data/Churn_Modelling.csv`. Encodes domain assumptions (e.g. higher age, Germany, and low activity → higher churn) so the model sees meaningful patterns during training.

- **`models/`** (runtime)  
  Created/populated at runtime with:
  - `best_model.joblib` – serialized best-performing classifier.
  - `preprocessor.joblib` – serialized `ChurnPreprocessor`.
  - `model_metadata.json` – metrics and metadata for the best model.  
  These are written by `ModelTrainer` and consumed by `ChurnService` to serve predictions.

- **`logs/churn_api.log`** (runtime)  
  Central application log file configured in `app/utils/logging.py`, capturing training events, model loading, and errors for observability.

---

### `app/` package (application code)

- **`app/__init__.py`**  
  Declares the package and exposes `__version__`. This version is used in metadata/health responses so clients can understand which build they are talking to.

- **`app/config.py`**  
  Central configuration using `pydantic-settings`. Defines:
  - App metadata (`app_name`, `debug`, `log_level`).
  - Standardized paths (`base_dir`, `models_dir`, `data_dir`, `logs_dir`).
  - Default filenames for persisted artifacts (`best_model.joblib`, `preprocessor.joblib`, `model_metadata.json`).
  - `supported_algorithms` list.  
  Design intent: single source of truth for paths and knobs, easily overridden from environment (e.g. via `CHURN_*` env vars).

- **`app/main.py`**  
  FastAPI entrypoint. Responsibilities:
  - Configure **lifespan**: on startup, set up logging, ensure folders exist, and eagerly initialize `ChurnService` so models are loaded if already trained.
  - Create `FastAPI` app with title, description, version, and docs URLs.
  - Apply **CORS** middleware to allow cross-origin access (useful for dashboards).
  - Include the API router under prefix `/api/v1`.
  - Register a global exception handler for `ChurnPredictionError` to consistently serialize custom errors as JSON.
  - Provide a root (`/`) endpoint that points to `/docs` and `/api/v1/health`.  
  Thinking: keep startup concerns, cross-cutting middleware, and error translation close to the framework boundary, away from the ML/business logic.

---

### `app/api/` (HTTP layer)

- **`app/api/__init__.py`**  
  Marks the `api` package; no runtime logic.

- **`app/api/schemas.py`**  
  All **Pydantic models** that define the HTTP contract:
  - `CustomerFeatures`: validates and documents input features for a single customer, including ranges and types. Contains an example payload for Swagger UI.
  - `SinglePredictionResponse`: output format for one prediction (label + probability + optional `customer_id`).
  - `BatchPredictionRequest` / `BatchPredictionResponse`: wrap lists of `CustomerFeatures` and corresponding predictions.
  - `TrainResponse`: response structure after training (best algorithm, score, metrics, all-results).
  - `ModelMetadataResponse`: shape of metadata served from `/metadata`.
  - `HealthResponse`: structure of `/health` response.  
  Thinking: centralizing schemas ensures that input validation and API documentation remain in sync with the backend logic and encourages type safety.

- **`app/api/routes.py`**  
  Defines all HTTP endpoints:
  - `GET /health`: calls `ChurnService.is_loaded()` and returns basic app health and version.
  - `GET /metadata`: calls `ChurnService.get_metadata()` to expose model details and metrics.
  - `POST /predict`: takes one `CustomerFeatures` object, calls `ChurnService.predict_single`, and returns a `SinglePredictionResponse`. Handles `ModelNotTrainedError` (503) and validation/preprocessing errors (422).
  - `POST /predict/batch`: accepts a list of customers and returns batched predictions.
  - `POST /train`: accepts an optional CSV (`UploadFile`) and optional `algorithms` and `metric_for_best` form fields. Writes the upload to a temp file or falls back to `data/Churn_Modelling.csv`, then calls `ChurnService.train` and returns a `TrainResponse`.  
  Design approach: keep these functions thin HTTP handlers that:
  1. Validate/parse request data into Pydantic models.
  2. Delegate business logic to `ChurnService`.
  3. Map domain errors to HTTP status codes.

---

### `app/services/` (service / domain layer)

- **`app/services/__init__.py`**  
  Exposes `ChurnService` at the package level for easier imports.

- **`app/services/churn_service.py`**  
  Core service that orchestrates **model lifecycle** and **business logic**:
  - Knows where models, preprocessors, and metadata live via `Settings`.
  - `_load_model_if_exists()`: loads persisted `best_model.joblib` and `preprocessor.joblib`, plus `model_metadata.json`, logging failures instead of crashing.
  - `is_loaded()`: simple readiness check used by health and routes before serving predictions.
  - `predict_single()`: converts one feature dict to a `pandas` row, applies the preprocessor, and asks the loaded model for prediction and probability.
  - `predict_batch()`: same as above but vectorized for many customers.
  - `train()`: reads CSV into `DataFrame`, fits a fresh `ChurnPreprocessor`, instantiates `ModelTrainer`, and triggers training + saving of best artifacts. Then reloads them into memory.
  - `get_metadata()`: returns on-disk metadata if present or constructs a minimal metadata dict otherwise.  
  Thinking: this file is the **single abstraction** the rest of the app depends on for training and prediction. All ML details (algorithms, preprocessing) are hidden behind this service, making it easier to swap implementations later.

---

### `app/core/` (ML core)

- **`app/core/__init__.py`**  
  Re-exports `ChurnPreprocessor`, `ModelTrainer`, and `evaluate_model` to present a simple public interface for core ML operations.

- **`app/core/preprocessing.py`**  
  Encapsulates all **tabular preprocessing**:
  - Defines `FEATURE_COLUMNS`, `NUMERIC_COLUMNS`, `CATEGORICAL_COLUMNS`, and `TARGET_COLUMN` for churn.
  - `ChurnPreprocessor` builds a `ColumnTransformer` that standard-scales numeric features and one-hot encodes categorical ones (dropping one category to avoid multicollinearity, with `handle_unknown="ignore"`).
  - `fit()`: validates that all expected feature columns are present, casts categoricals to string, fits the transformer, and records feature names.
  - `transform()`: applies the fitted transformer to new data; raises `PreprocessingError` if called before `fit()`.
  - `save()` / `load()`: persist and restore the preprocessor using `joblib`.  
  Design intent: make preprocessing **reproducible and coupled** with the model by serializing the exact fitted transformer, eliminating “train/test preprocessing drift”.

- **`app/core/models.py`**  
  Implements **model training and selection**:
  - `ALGORITHMS`: dictionary mapping algorithm names to instantiated classifiers (Logistic Regression, Random Forest, XGBoost), all configured for balanced classes and a reasonable starting configuration.
  - `ModelTrainer`:
    - Accepts a fitted `ChurnPreprocessor`, an output directory, and a metric name (default F1).
    - `train(df, algorithms=None)`:  
      1. Uses the preprocessor to convert `df[FEATURE_COLUMNS]` into `X`.  
      2. Extracts `y = df[TARGET_COLUMN]`.  
      3. Iterates through chosen algorithms, fitting each model, computing predictions and probabilities, and calling `evaluate_model`.  
      4. Tracks the **best model** by `metric_for_best` (falls back to accuracy if needed).  
      5. Persists the best estimator and the preprocessor, and writes rich metadata to `model_metadata.json`.  
  Thinking: centralize all model experimentation and selection logic in one place so that adding models, tuning, or changing the selection metric does not affect API or service layers.

- **`app/core/evaluation.py`**  
  Houses the evaluation logic:
  - `evaluate_model(y_true, y_pred, y_prob)`: calculates accuracy, precision, recall, F1, ROC-AUC (when possible), and confusion matrix, and builds a `classification_report` dict.  
  Reasoning: keeping metric computation isolated makes it easy to extend (e.g. add PR-AUC, log-loss) while letting `ModelTrainer` stay focused on orchestration.

---

### `app/utils/` (cross-cutting utilities)

- **`app/utils/__init__.py`**  
  Exposes `get_logger` and `setup_logging` for convenient imports.

- **`app/utils/logging.py`**  
  Central logging configuration:
  - `setup_logging()`: configures a root `app` logger with a structured format, attaches console and optional file handlers, and respects `log_level`.  
  - `get_logger(name)`: returns a child logger with a consistent namespace (`app.<module>`).  
  Design intent: make logging configuration one-time and consistent across modules while enabling redirection to file in production.

- **`app/utils/exceptions.py`**  
  Defines domain-specific exceptions:
  - `ChurnPredictionError` (base class).
  - `ModelNotTrainedError`, `InvalidInputError`, `TrainingError`, `PreprocessingError`.  
  These are thrown in core/service layers and converted to HTTP responses by routes or the global exception handler in `app/main.py`. This gives clear, typed failure modes instead of ad-hoc `ValueError`s.

---

### Flow Summary (How Files Work Together)

1. **App startup**  
   - Docker or local process runs `uvicorn app.main:app`.  
   - `lifespan` in `app/main.py` initializes logging (via `app/utils/logging.py`), ensures directories from `app/config.py` exist, and instantiates `ChurnService` from `app/services/churn_service.py`, which attempts to load any existing model/preprocessor/metadata from `models/`.

2. **Training (`POST /api/v1/train`)**  
   - Request hits FastAPI and is routed by `app/api/routes.py`.  
   - File and form fields are validated and written to a temp CSV or resolved to `data/Churn_Modelling.csv`.  
   - `ChurnService.train()` is called; it uses `ChurnPreprocessor` (`app/core/preprocessing.py`) and `ModelTrainer` (`app/core/models.py`) to fit all configured algorithms, evaluate them via `evaluate_model` (`app/core/evaluation.py`), and persist the best model and metadata into `models/`.  
   - Metadata is exposed to clients and logged to `logs/churn_api.log`.

3. **Prediction (`POST /api/v1/predict` / `/predict/batch`)**  
   - Requests are parsed into Pydantic models in `app/api/schemas.py`.  
   - Routes delegate to `ChurnService.predict_single` / `predict_batch`.  
   - These use the loaded preprocessor and model to transform features and compute predictions and probabilities.  
   - Responses are serialized back into schema types and returned over HTTP.

4. **Health & metadata**  
   - `GET /api/v1/health` and `GET /api/v1/metadata` expose readiness, basic status, and detailed training results by querying `ChurnService` and reading `model_metadata.json`.

Overall design thinking: **each file has a single, clear responsibility** (HTTP layer, service layer, ML core, utilities, config), making the project easy to understand, extend, and maintain while keeping the training–serving flow explicit and testable.

