# Bank Customer Churn Prediction System

Production-ready API to predict whether a bank customer is likely to churn based on credit score, geography, gender, age, tenure, balance, number of products, credit card status, activity status, and estimated salary.

## Features

- **Data preprocessing**: Standard scaling for numerics, one-hot encoding for Geography and Gender
- **Multiple algorithms**: Logistic Regression, Random Forest, XGBoost
- **Evaluation**: Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, classification report
- **Best model selection**: Automatically selects and persists the best model by chosen metric (default: F1)
- **REST API**:
  - `GET /api/v1/health` — Health check
  - `POST /api/v1/train` — Train models (CSV upload or default dataset)
  - `POST /api/v1/predict` — Single customer prediction
  - `POST /api/v1/predict/batch` — Batch predictions
  - `GET /api/v1/metadata` — Model metadata and metrics
- **Logging**, **error handling**, and **Docker** support

## Project structure

```
bank-churn/
├── app/
│   ├── main.py           # FastAPI app
│   ├── config.py         # Settings
│   ├── api/
│   │   ├── routes.py     # API routes
│   │   └── schemas.py    # Pydantic schemas
│   ├── core/
│   │   ├── preprocessing.py
│   │   ├── models.py     # Training & best-model selection
│   │   └── evaluation.py
│   ├── services/
│   │   └── churn_service.py
│   └── utils/
│       ├── logging.py
│       └── exceptions.py
├── data/
│   └── Churn_Modelling.csv   # Sample training data
├── models/                    # Saved model, preprocessor, metadata (created at runtime)
├── logs/                      # Application logs
├── scripts/
│   └── generate_sample_data.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Quick start

### Local (no Docker)

1. Create a virtual environment and install dependencies:

   ```bash
   cd bank-churn
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```

2. Train a model (uses `data/Churn_Modelling.csv` if no file is uploaded):

   ```bash
   curl -X POST http://localhost:8000/api/v1/train
   ```

   Or with a custom CSV:

   ```bash
   curl -X POST -F "file=@path/to/your.csv" http://localhost:8000/api/v1/train
   ```

   Optional form fields: `algorithms=logistic_regression,random_forest,xgboost`, `metric_for_best=f1_score`.

3. Start the API:

   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

4. Open http://localhost:8000/docs for Swagger UI. Try:
   - **GET /api/v1/health**
   - **GET /api/v1/metadata**
   - **POST /api/v1/predict** with a JSON body (see schema in docs).

## Backend Swagger API

Here is the latest snapshot of our backend Swagger documentation:

![Backend Swagger](result_images/backend_swagger.png)

### Docker

```bash
cd bank-churn
docker-compose up --build
```

Then train (if needed) and call the same endpoints as above on port 8000. Models and logs are persisted in `./models` and `./logs` via volumes.

## CSV format for training

Required columns: `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`, `Exited` (0/1). Extra columns (e.g. `RowNumber`, `CustomerId`, `Surname`) are ignored.

## Environment variables

- `CHURN_LOG_LEVEL` — Log level (default: INFO)
- `CHURN_DEBUG` — Enable debug mode


