# Bank Customer Churn Prediction — Project Details

This document describes the project from planning to execution: design choices, models, tech stack, flow, and how to extend the system (new models, LLM integration).

---

## 1. Plan to Execution

### 1.1 Planning Phase

| Step | Decision |
|------|----------|
| **Objective** | Predict binary churn (0 = stay, 1 = leave) from structured customer data. |
| **Inputs** | Credit score, geography, gender, age, tenure, balance, number of products, credit card, activity, estimated salary. |
| **Output** | Prediction (0/1) and churn probability for single or batch requests. |
| **Delivery** | REST API with train-once, predict-many; Docker for deployment. |
| **Quality** | Standard classification metrics; automatic best-model selection; preprocessing pipeline persisted with the model. |

### 1.2 Execution Phases

1. **Structure** — Layered app: `config`, `core` (preprocessing, models, evaluation), `api` (routes, schemas), `services`, `utils`.
2. **Preprocessing** — Define feature/target columns; build `ChurnPreprocessor` (numeric scaling + categorical encoding); make it serializable with the model.
3. **Models** — Implement `ModelTrainer` with multiple algorithms; evaluate each; select best by a chosen metric (default F1); save best model + preprocessor + metadata.
4. **API** — FastAPI app with health, metadata, train, single predict, batch predict; Pydantic schemas; clear error handling.
5. **Service layer** — `ChurnService` owns model/preprocessor load/save and predict/train; single place for business logic.
6. **Ops** — Logging, custom exceptions, Dockerfile, docker-compose, README, sample data.

---

## 2. Models Used & Thought Process

### 2.1 Algorithms Implemented

| Model | File (definition) | Rationale |
|-------|-------------------|-----------|
| **Logistic Regression** | `app/core/models.py` → `ALGORITHMS["logistic_regression"]` | Simple, interpretable baseline; good for binary classification; `class_weight="balanced"` to handle imbalanced churn. |
| **Random Forest** | `app/core/models.py` → `ALGORITHMS["random_forest"]` | Handles non-linearity and feature interactions without heavy tuning; robust to outliers; same class balancing. |
| **XGBoost** | `app/core/models.py` → `ALGORITHMS["xgboost"]` | Strong performance on tabular data; gradient boosting; often best for churn-style problems. |

### 2.2 Design Choices

- **Metric for “best” model**: Default is **F1** so both precision and recall matter (important when churn is minority class).
- **Class balancing**: All three models use balancing (e.g. `class_weight="balanced"` or equivalent) to avoid predicting “no churn” always.
- **No cross-validation in selection**: Current flow trains once on full data and picks best by in-sample metric. Cross-validation can be added later for more robust selection (see “Scope of improvement”).
- **Single best model**: Only one model (the best by chosen metric) is saved and served; the rest are used only during training for comparison.

---

## 3. Tech Stack & Why

| Component | Choice | Reason |
|-----------|--------|--------|
| **Language** | Python 3.11 | Standard for ML; rich ecosystem (sklearn, pandas, XGBoost). |
| **API** | FastAPI | Async, automatic OpenAPI docs, Pydantic validation, good performance. |
| **ML** | scikit-learn, XGBoost | Proven, production-friendly; joblib for serialization. |
| **Data** | pandas, numpy | Standard for tabular preprocessing and feature matrices. |
| **Config** | pydantic-settings | Type-safe config with env vars and `.env` support. |
| **Serialization** | joblib | Native for sklearn/XGBoost; one file per model/preprocessor. |
| **Deployment** | Docker + docker-compose | Reproducible runs; volumes for `models/`, `logs/`, `data/`. |

### Advantages of This Stack

- **Clear separation**: Config, core ML, API, and service layer are separate; easy to test and change one part.
- **Production-oriented**: Logging, structured errors, health and metadata endpoints, Docker.
- **Extensible**: New models = add to `ALGORITHMS`; new endpoints = add routes; new features = extend preprocessing.
- **Documented API**: `/docs` and `/redoc` from FastAPI; Pydantic schemas document request/response.

---

## 4. Scope of Improvement

- **Evaluation**: Add **cross-validation** (e.g. 5-fold) when selecting the best model; report mean ± std of the chosen metric in metadata.
- **Hyperparameter tuning**: Add **GridSearchCV / RandomizedSearchCV** (or Optuna) for each algorithm; persist best params in metadata.
- **Data**: Support **train/validation/test** split; report test-set metrics in metadata; optional **feature selection** (e.g. RF feature importance, recursive feature elimination).
- **Preprocessing**: **Missing value** handling (imputation or “missing” category); **outlier** capping; optional **SMOTE/oversampling** for severe imbalance.
- **API**: **Versioning** (e.g. `/api/v2`) for breaking changes; **rate limiting**; **auth** (API key or JWT) for `/train` and `/predict`.
- **Model**: **A/B or shadow deployment** (e.g. run two models, log both outputs, switch later); **model versioning** (e.g. by timestamp or git commit).
- **Monitoring**: **Prometheus metrics** (request count, latency, prediction distribution); **alerting** on health or metric drift.

---

## 5. Adding More Models — Where & How

### 5.1 Files to Touch

| Goal | Primary file | Other files (if needed) |
|------|-----------------------------|--------------------------|
| Add a new algorithm | `app/core/models.py` | `app/config.py` (optional: add to `supported_algorithms`) |
| Change selection metric | `app/core/models.py` (`ModelTrainer`), `app/services/churn_service.py` (train call), `app/api/routes.py` (form param) | — |
| Add hyperparameter config | `app/core/models.py` (and optionally `app/config.py` or env) | — |

### 5.2 Step-by-Step: Add a New Model (e.g. LightGBM)

1. **Dependency**  
   Add to `requirements.txt`:
   ```text
   lightgbm==4.2.0
   ```

2. **Register algorithm** in `app/core/models.py`:
   - Import: `import lightgbm as lgb`
   - Add to `ALGORITHMS`:
   ```python
   "lightgbm": lgb.LGBMClassifier(
       n_estimators=100,
       random_state=42,
       class_weight="balanced",
       verbose=-1,
   ),
   ```

3. **Optional: config**  
   In `app/config.py`, add `"lightgbm"` to `supported_algorithms` so docs and validation stay in sync.

4. **Training**  
   No change needed: `ModelTrainer.train()` already iterates over `ALGORITHMS` (or the `algorithms` list passed in). New model is included automatically; if it wins on the chosen metric, it becomes the saved “best” model.

5. **Serving**  
   No change: `ChurnService` loads whatever is in `best_model.joblib`; the new model is used if it was selected during the last training run.

**Summary**: Only `app/core/models.py` (and optionally `app/config.py`) need edits to introduce a new model. The rest of the flow (preprocessing → train → evaluate → save best → serve) stays the same.

---

## 6. Extending to LLM — Options & Flow

### 6.1 What “LLM” Could Mean Here

- **A) Explainability**: Use an LLM to generate natural-language explanations of why a customer was predicted to churn (e.g. from feature values + model metadata).
- **B) Hybrid input**: Add free-text (e.g. “reason for last call”) and use an LLM to embed or classify it; combine with existing tabular model (ensemble or feature).
- **C) Full LLM classifier**: Replace or complement the tabular model with an LLM that takes structured + optional text and outputs churn prediction (e.g. fine-tuned or prompt-based).

### 6.2 Where to Integrate (Files & Flow)

| Approach | Where it fits | New/Modified components |
|----------|----------------|---------------------------|
| **A) Explanations** | After prediction in the API | New module e.g. `app/core/explainer.py`; call from `app/services/churn_service.py` or from `app/api/routes.py`; add field to prediction response. |
| **B) Hybrid (text + table)** | Preprocessing + model | New preprocessing step for text (embedding or LLM-derived features) in `app/core/preprocessing.py` or a separate `app/core/text_features.py`; extend feature list; optionally new model in `app/core/models.py` that uses extended features. |
| **C) LLM as classifier** | Replace or parallel to current trainer | New `app/core/llm_predictor.py` (prompt or fine-tuned); `ChurnService` loads either classic model or LLM (or both); new/updated routes for “explain” or “predict with LLM”. |

### 6.3 Suggested Flow for “Explainability” (A)

1. **New module** `app/core/explainer.py`:
   - Input: customer features (dict), prediction (0/1), probability, optional model name.
   - Use a small LLM (local or API) with a prompt like: “Given these customer attributes and churn prediction, list 2–3 short reasons why they might churn.”
   - Return a string or list of reasons.

2. **Service** `app/services/churn_service.py`:
   - Add optional `explain: bool` to `predict_single` (and optionally batch).
   - If `explain`, after predicting, call the explainer and attach the result to the response.

3. **API** `app/api/routes.py` and `app/api/schemas.py`:
   - Add optional query/body param `explain: bool`.
   - Extend `SinglePredictionResponse` with `explanation: Optional[str]`.
   - Return explanation when requested.

4. **Config** `app/config.py`:
   - Add settings for LLM (e.g. provider, model name, API key env var).

### 6.4 Suggested Flow for “Hybrid Text + Table” (B)

1. **Text pipeline** in `app/core/text_features.py` (or inside preprocessing):
   - Accept optional text per customer (e.g. “last complaint”).
   - Use an embedding model (e.g. sentence-transformers) or an LLM to get a fixed-size vector (or a single “sentiment/risk” score).
   - Output numeric features to be concatenated with tabular features.

2. **Preprocessing** `app/core/preprocessing.py`:
   - Extend `FEATURE_COLUMNS` (or add a separate path) for text-derived features.
   - In `fit`/`transform`, call the text pipeline when text is present; otherwise use zeros or a “missing” embedding.

3. **Training**:
   - Training data must include the new text-derived columns (or a raw text column that the pipeline converts).
   - `ModelTrainer` and existing models work as-is on the extended feature matrix.

4. **API**:
   - Add optional text field(s) to `CustomerFeatures` (or a separate “context” object).
   - In `ChurnService.predict_single`/`predict_batch`, run the text pipeline and merge features before calling the model.

### 6.5 Dependencies for LLM

- For **explanations** or **API-based LLM**: `openai`, `httpx`, or provider SDK; optional `langchain` for prompts.
- For **local embeddings/text**: `sentence-transformers` or `transformers` + `torch`.
- Add these to `requirements.txt` and document in README; keep LLM code behind feature flags or config so the core churn pipeline stays optional.

---

## 7. Full Flow of the Project

### 7.1 High-Level Data & Control Flow

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BANK CHURN PREDICTION FLOW                         │
└─────────────────────────────────────────────────────────────────────────────┘

  TRAINING (one-time or on-demand via POST /train)
  ───────────────────────────────────────────────
  CSV (data/Churn_Modelling.csv or upload)
       │
       ▼
  ┌─────────────────┐     ┌──────────────────────────────────────────────┐
  │ ChurnPreprocessor│     │ FEATURE_COLUMNS → numeric scaling + one-hot   │
  │ .fit(df)         │     │ TARGET_COLUMN = "Exited"                      │
  └────────┬─────────┘     └──────────────────────────────────────────────┘
           │
           ▼
  X (numeric matrix), y (binary)
           │
           ▼
  ┌─────────────────────────────────────────────────────────────────────────┐
  │ ModelTrainer.train(df, algorithms=[...])                                 │
  │   For each algorithm (LogisticRegression, RandomForest, XGBoost):       │
  │     fit(X, y) → predict(X) → predict_proba(X) → evaluate_model(...)      │
  │   Select best by metric (default: f1_score)                             │
  │   Save: best_model.joblib, preprocessor.joblib, model_metadata.json     │
  └─────────────────────────────────────────────────────────────────────────┘
           │
           ▼
  models/ populated; ChurnService._load_model_if_exists() loads them

  PREDICTION (single or batch)
  ───────────────────────────
  Request (JSON: customer features)
       │
       ▼
  ChurnService.predict_single(features) or .predict_batch(list)
       │
       ▼
  ChurnPreprocessor.transform(df_from_features)
       │
       ▼
  model.predict(X), model.predict_proba(X)
       │
       ▼
  Response: churn_prediction (0/1), churn_probability
```

### 7.2 Request Flow Through the Codebase

| Step | Component | File |
|------|-----------|------|
| 1. HTTP request | FastAPI | `app/main.py` |
| 2. Routing | Router | `app/api/routes.py` |
| 3. Validation | Pydantic | `app/api/schemas.py` |
| 4. Business logic | ChurnService | `app/services/churn_service.py` |
| 5. Preprocessing | ChurnPreprocessor | `app/core/preprocessing.py` |
| 6. Model inference | Loaded estimator (joblib) | — |
| 7. Response | Pydantic | `app/api/schemas.py` → `app/api/routes.py` |

### 7.3 File Map (What Lives Where)

| Path | Purpose |
|------|---------|
| `app/main.py` | FastAPI app, lifespan, CORS, exception handler, mount router. |
| `app/config.py` | Settings (paths, model filenames, supported algorithms). |
| `app/api/routes.py` | Health, metadata, train, single predict, batch predict. |
| `app/api/schemas.py` | Request/response models (CustomerFeatures, TrainResponse, etc.). |
| `app/services/churn_service.py` | Load/save model and preprocessor; predict_single/batch; train; metadata. |
| `app/core/models.py` | ALGORITHMS dict, ModelTrainer (train, evaluate, save best). |
| `app/core/preprocessing.py` | FEATURE_COLUMNS, TARGET_COLUMN, ChurnPreprocessor (fit/transform/save/load). |
| `app/core/evaluation.py` | evaluate_model (accuracy, precision, recall, F1, ROC-AUC, confusion matrix). |
| `app/utils/logging.py` | setup_logging, get_logger. |
| `app/utils/exceptions.py` | ChurnPredictionError, ModelNotTrainedError, InvalidInputError, TrainingError, PreprocessingError. |
| `data/Churn_Modelling.csv` | Sample training data. |
| `models/` | best_model.joblib, preprocessor.joblib, model_metadata.json (at runtime). |
| `scripts/generate_sample_data.py` | Optional script to generate larger synthetic CSV. |
| `Dockerfile`, `docker-compose.yml` | Container build and run; volumes for models, logs, data. |

### 7.4 API Endpoints Summary

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/v1/health` | Liveness/readiness; version; whether a model is loaded. |
| GET | `/api/v1/metadata` | Best algorithm, metric, scores, feature/target columns. |
| POST | `/api/v1/train` | Train from default CSV or uploaded file; form: `file`, `algorithms`, `metric_for_best`. |
| POST | `/api/v1/predict` | Single prediction; body: CustomerFeatures JSON. |
| POST | `/api/v1/predict/batch` | Batch prediction; body: list of CustomerFeatures. |

---

This document is the single source of truth for project plan, execution, models, stack, advantages, improvements, how to add models, how to extend to LLM, and full project flow.
