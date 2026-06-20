import sys
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.pipeline_utils import (
    clean_telco_data,
    engineer_features,
    assign_risk_level,
    observed_risk_factors,
    mask_customer_id
)


MODEL_PATH = (
    BASE_DIR
    / "models"
    / "churn_risk_pipeline.joblib"
)

METADATA_PATH = (
    BASE_DIR
    / "models"
    / "model_metadata.json"
)

DEFAULT_DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "cleaned_churn_data.csv"
)


st.set_page_config(
    page_title="Customer Churn Risk MVP",
    layout="wide"
)

st.title("Telecommunication Customer Churn Risk MVP")

st.caption(
    "Internal proof of concept for customer-retention prioritization. "
    "Predictions are decision-support estimates and do not guarantee "
    "that a customer will churn."
)


@st.cache_resource
def load_model_assets():
    model = joblib.load(MODEL_PATH)

    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )

    return model, metadata


@st.cache_data
def load_default_data():
    return pd.read_csv(DEFAULT_DATA_PATH)


model, metadata = load_model_assets()

st.sidebar.header("Data Source")

uploaded_file = st.sidebar.file_uploader(
    "Upload an authorized Telco CSV",
    type=["csv"]
)

if uploaded_file is not None:
    source_df = pd.read_csv(uploaded_file)
    source_name = uploaded_file.name
else:
    source_df = load_default_data()
    source_name = "Bundled cleaned classroom dataset"

try:
    cleaned_df, audit = clean_telco_data(
        source_df,
        require_target=False
    )

    cleaned_df = (
        cleaned_df
        .drop_duplicates(
            subset=["customerID"],
            keep="first"
        )
        .reset_index(drop=True)
    )

    engineered_df = engineer_features(cleaned_df)

except Exception as error:
    st.error(f"Dataset validation failed: {error}")
    st.stop()

model_features = metadata["model_features"]

missing_model_features = [
    column for column in model_features
    if column not in engineered_df.columns
]

if missing_model_features:
    st.error(
        "The scoring data is missing model features: "
        + ", ".join(missing_model_features)
    )
    st.stop()

probabilities = model.predict_proba(
    engineered_df[model_features]
)[:, 1]

monthly_charge_reference = float(
    engineered_df["MonthlyCharges"].median()
)

risk_df = engineered_df[[
    "customerID",
    "tenure",
    "TenureBand",
    "Contract",
    "PaymentMethod",
    "MonthlyCharges",
    "ServiceCount",
    "TechSupport",
    "OnlineSecurity",
    "InternetService"
]].copy()

risk_df["CustomerReference"] = (
    risk_df["customerID"]
    .map(mask_customer_id)
)

risk_df["ChurnProbability"] = probabilities

risk_df["RiskLevel"] = [
    assign_risk_level(
        probability,
        low_threshold=metadata["low_risk_threshold"],
        high_threshold=metadata["high_risk_threshold"]
    )
    for probability in probabilities
]

risk_df["ObservedRiskFactors"] = (
    risk_df.apply(
        lambda row: observed_risk_factors(
            row,
            monthly_charge_reference=monthly_charge_reference
        ),
        axis=1
    )
)

risk_df = risk_df.sort_values(
    "ChurnProbability",
    ascending=False
).reset_index(drop=True)

st.sidebar.header("Filter Options")

risk_options = ["Low", "Medium", "High"]

selected_risks = st.sidebar.multiselect(
    "Risk Level",
    options=risk_options,
    default=risk_options
)

contract_options = sorted(
    risk_df["Contract"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

selected_contracts = st.sidebar.multiselect(
    "Contract Type",
    options=contract_options,
    default=contract_options
)

tenure_options = sorted(
    risk_df["TenureBand"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

selected_tenure_groups = st.sidebar.multiselect(
    "Tenure Group",
    options=tenure_options,
    default=tenure_options
)

search_term = st.sidebar.text_input(
    "Search masked customer reference"
)

filtered_df = risk_df[
    risk_df["RiskLevel"].isin(selected_risks)
    & risk_df["Contract"].astype(str).isin(selected_contracts)
    & risk_df["TenureBand"].astype(str).isin(
        selected_tenure_groups
    )
].copy()

if search_term:
    filtered_df = filtered_df[
        filtered_df["CustomerReference"]
        .str.contains(
            search_term,
            case=False,
            na=False
        )
    ]

metric_1, metric_2, metric_3, metric_4 = st.columns(4)

metric_1.metric(
    "Scored Customers",
    f"{len(risk_df):,}"
)

metric_2.metric(
    "High-Risk Customers",
    f"{(risk_df['RiskLevel'] == 'High').sum():,}"
)

metric_3.metric(
    "Average Predicted Risk",
    f"{risk_df['ChurnProbability'].mean():.1%}"
)

metric_4.metric(
    "Displayed Records",
    f"{len(filtered_df):,}"
)

st.caption(
    f"Data source: {source_name} | "
    f"Selected model: {metadata['selected_model']} | "
    f"Model created: {metadata['created_at']}"
)

st.subheader("Filtered Customer Risk Table")

display_columns = [
    "CustomerReference",
    "TenureBand",
    "Contract",
    "MonthlyCharges",
    "ServiceCount",
    "ChurnProbability",
    "RiskLevel",
    "ObservedRiskFactors"
]

st.dataframe(
    filtered_df[display_columns],
    use_container_width=True,
    hide_index=True,
    column_config={
        "ChurnProbability": st.column_config.ProgressColumn(
            "Churn Probability",
            min_value=0.0,
            max_value=1.0,
            format="percent"
        )
    }
)

download_df = filtered_df[display_columns].copy()

download_csv = download_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    "Download Filtered Risk List",
    data=download_csv,
    file_name="filtered_customer_churn_risk.csv",
    mime="text/csv"
)

st.subheader("Risk Distribution")

risk_distribution = (
    filtered_df["RiskLevel"]
    .value_counts()
    .reindex(["Low", "Medium", "High"])
    .fillna(0)
)

st.bar_chart(risk_distribution)

st.subheader("Average Risk by Contract")

contract_risk = (
    filtered_df
    .groupby("Contract")["ChurnProbability"]
    .mean()
    .sort_values(ascending=False)
)

st.bar_chart(contract_risk)

st.subheader("Individual Customer Review")

customer_options = (
    filtered_df["CustomerReference"]
    .dropna()
    .tolist()
)

if customer_options:
    selected_customer = st.selectbox(
        "Select a masked customer reference",
        options=customer_options
    )

    customer_row = filtered_df[
        filtered_df["CustomerReference"]
        == selected_customer
    ].iloc[0]

    detail_1, detail_2, detail_3 = st.columns(3)

    detail_1.metric(
        "Risk Level",
        customer_row["RiskLevel"]
    )

    detail_2.metric(
        "Predicted Probability",
        f"{customer_row['ChurnProbability']:.1%}"
    )

    detail_3.metric(
        "Tenure",
        f"{int(customer_row['tenure'])} months"
    )

    st.write(
        "**Observed risk factors:**",
        customer_row["ObservedRiskFactors"]
    )

    st.info(
        "Observed risk factors are rule-based explanations for review. "
        "They are associations and should not be interpreted as proven "
        "causes of churn."
    )

else:
    st.warning(
        "No customers match the selected filters."
    )

with st.expander("Data Cleaning Audit"):
    st.json(audit)

with st.expander("Model and Threshold Information"):
    st.json(metadata)
