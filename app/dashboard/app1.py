import pathlib
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import shap
import streamlit as st
from matplotlib import pyplot as plt

# Ensure project root (which contains the `app` package) is on sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.config import get_settings
from app.core.preprocessing import ChurnPreprocessor, FEATURE_COLUMNS, CATEGORICAL_COLUMNS


@st.cache_resource
def load_data() -> pd.DataFrame:
    settings = get_settings()
    path = settings.data_dir / "Bank Customer Churn Prediction.csv" #data\Bank Customer Churn Prediction.csv
    return pd.read_csv(path)


@st.cache_resource
def load_model_and_preprocessor():
    settings = get_settings()
    model_path = settings.models_dir / settings.default_model_filename
    preprocessor_path = settings.models_dir / settings.default_preprocessor_filename
    if not model_path.exists() or not preprocessor_path.exists():
        return None, None
    model = joblib.load(model_path)
    preprocessor = ChurnPreprocessor.load(preprocessor_path)
    return model, preprocessor


def page_eda(df: pd.DataFrame) -> None:
    st.header("EDA Dashboard")
    st.markdown("Quick overview of the churn dataset with interactive charts.")

    st.subheader("Target distribution (Exited)")
    target_counts = df["Exited"].value_counts().rename(index={0: "Stay", 1: "Churn"})
    st.bar_chart(target_counts)

    st.subheader("Numerical feature distributions")
    numeric_cols = ["CreditScore", "Age", "Balance", "EstimatedSalary"]
    col = st.selectbox("Select numeric feature", numeric_cols)
    fig = px.histogram(df, x=col, color="Exited", marginal="box", nbins=40, barmode="overlay")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation with churn")
    corr = df[["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary", "Exited"]].corr()
    fig_corr, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(corr, annot=True, cmap="Blues", ax=ax)
    st.pyplot(fig_corr)


def page_feature_analysis(df: pd.DataFrame) -> None:
    st.header("Feature Analysis")
    st.markdown("Explore how individual features relate to churn.")

    # Geography / Gender vs churn
    st.subheader("Churn rate by Geography and Gender")
    group_cols = ["Geography", "Gender"]
    grouped = df.groupby(group_cols)["Exited"].mean().reset_index()
    grouped["ChurnRate"] = grouped["Exited"]
    fig = px.bar(grouped, x="Geography", y="ChurnRate", color="Gender", barmode="group", text_auto=".2f")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Balance vs Age (colored by churn)")
    fig_scatter = px.scatter(
        df.sample(min(len(df), 2000), random_state=42),
        x="Age",
        y="Balance",
        color="Exited",
        opacity=0.7,
        labels={"Exited": "Churn"},
    )
    st.plotly_chart(fig_scatter, use_container_width=True)


def page_feature_importance(df: pd.DataFrame, model, preprocessor: ChurnPreprocessor | None) -> None:
    st.header("Model & Feature Importance")
    if model is None or preprocessor is None:
        st.warning("Train a model first (via API /train) to see feature importance.")
        return

    X = preprocessor.transform(df)
    feature_names = preprocessor.feature_names_out

    st.subheader("Model type")
    st.write(type(model).__name__)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False).head(20)
        fig = px.bar(imp_df, x="importance", y="feature", orientation="h")
        st.subheader("Top feature importances")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Current best model does not expose `feature_importances_`. Showing SHAP summary if possible.")

    try:
        st.subheader("SHAP summary (top features)")
        # use a sample for speed
        sample_X = X[:1000]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(sample_X)
        fig_shap = shap.plots.beeswarm(shap_values, show=False)
        st.pyplot(fig_shap, clear_figure=True)
    except Exception as e:
        st.warning(f"Could not compute SHAP values: {e}")


def page_churn_prediction(model, preprocessor: ChurnPreprocessor | None) -> None:
    st.header("Churn Prediction")
    st.markdown("Interactively predict churn probability for a new customer.")

    if model is None or preprocessor is None:
        st.warning("Train a model first (via API /train) so the dashboard can load it.")
        return

    with st.form("churn_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            credit_score = st.slider("Credit Score", 300, 900, 650)
            age = st.slider("Age", 18, 92, 40)
            gender = st.selectbox("Gender", ["Male", "Female"])
        with col2:
            balance = st.number_input("Account Balance", 0.0, 300000.0, 60000.0, step=1000.0)
            estimated_salary = st.number_input("Estimated Salary", 0.0, 300000.0, 50000.0, step=1000.0)
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        with col3:
            tenure = st.slider("Tenure (years)", 0, 10, 3)
            num_products = st.slider("Number of Products", 1, 4, 2)
            has_cr_card = st.selectbox("Has Credit Card", ["No", "Yes"])
            is_active = st.selectbox("Is Active Member", ["No", "Yes"])

        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        row = {
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": 1 if has_cr_card == "Yes" else 0,
            "IsActiveMember": 1 if is_active == "Yes" else 0,
            "EstimatedSalary": estimated_salary,
        }
        df_row = pd.DataFrame([row])
        X = preprocessor.transform(df_row)
        proba = float(model.predict_proba(X)[0, 1])
        pred = int(model.predict(X)[0])

        st.success(f"Churn probability: **{proba:.3f}**  (prediction: {'Churn' if pred == 1 else 'Stay'})")


def main() -> None:
    st.set_page_config(page_title="Bank Churn – Analytics v2", layout="wide")
    st.title("Bank Customer Churn – Analytics Dashboard (v2)")

    df = load_data()
    model, preprocessor = load_model_and_preprocessor()

    menu = st.sidebar.radio(
        "Navigation",
        [
            "EDA Dashboard",
            "Feature Analysis",
            "Model & Feature Importance",
            "Churn Prediction",
        ],
    )

    if menu == "EDA Dashboard":
        page_eda(df)
    elif menu == "Feature Analysis":
        page_feature_analysis(df)
    elif menu == "Model & Feature Importance":
        page_feature_importance(df, model, preprocessor)
    elif menu == "Churn Prediction":
        page_churn_prediction(model, preprocessor)


if __name__ == "__main__":
    main()

