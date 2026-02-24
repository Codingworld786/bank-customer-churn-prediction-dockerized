"""Generate a sample Churn_Modelling dataset for training (synthetic)."""
import pandas as pd
import numpy as np
from pathlib import Path

# Columns expected by the app
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
TARGET = "Exited"
GEO = ["France", "Germany", "Spain"]
GENDER = ["Male", "Female"]


def main() -> None:
    np.random.seed(42)
    n = 2000
    df = pd.DataFrame({
        "RowNumber": np.arange(1, n + 1),
        "CustomerId": np.arange(100001, 100001 + n),
        "Surname": [f"Surname_{i}" for i in range(n)],
        "CreditScore": np.clip(np.random.normal(650, 100, n), 350, 850).astype(int),
        "Geography": np.random.choice(GEO, n),
        "Gender": np.random.choice(GENDER, n),
        "Age": np.clip(np.random.normal(39, 10, n), 18, 92).astype(int),
        "Tenure": np.random.randint(0, 11, n),
        "Balance": np.round(np.random.uniform(0, 250000, n), 2),
        "NumOfProducts": np.random.choice([1, 2, 3, 4], n, p=[0.2, 0.5, 0.25, 0.05]),
        "HasCrCard": np.random.randint(0, 2, n),
        "IsActiveMember": np.random.randint(0, 2, n),
        "EstimatedSalary": np.round(np.random.uniform(11, 200000, n), 2),
    })
    # Simple synthetic churn: higher age + Germany + low activity -> more churn
    exit_prob = (
        0.01 * df["Age"]
        + (df["Geography"] == "Germany").astype(int) * 0.15
        + (1 - df["IsActiveMember"]) * 0.1
        + np.random.uniform(0, 0.2, n)
    )
    df[TARGET] = (np.random.uniform(0, 1, n) < np.clip(exit_prob, 0, 1)).astype(int)

    out = Path(__file__).resolve().parent.parent / "data" / "Churn_Modelling.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")


if __name__ == "__main__":
    main()
