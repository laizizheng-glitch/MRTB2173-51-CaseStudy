import sys
import json
from pathlib import Path

import joblib
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline_utils import (
    clean_telco_data,
    engineer_features,
    assign_risk_level,
    validate_required_columns
)


def create_sample_dataframe():
    return pd.DataFrame([
        {
            "customerID": "TEST-001",
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 60.0,
            "TotalCharges": 60.0,
            "Churn": "Yes"
        }
    ])


def test_required_column_validation():
    sample = create_sample_dataframe()
    validate_required_columns(sample, require_target=True)

    incomplete = sample.drop(columns=["Contract"])

    with pytest.raises(ValueError):
        validate_required_columns(
            incomplete,
            require_target=True
        )


def test_exact_duplicate_rows_are_removed():
    sample = create_sample_dataframe()
    duplicated = pd.concat(
        [sample, sample],
        ignore_index=True
    )

    cleaned, audit = clean_telco_data(
        duplicated,
        require_target=True
    )

    assert len(cleaned) == 1
    assert audit["Exact duplicate rows removed"] == 1


def test_risk_levels_are_valid():
    probabilities = [0.10, 0.45, 0.80]

    levels = [
        assign_risk_level(value)
        for value in probabilities
    ]

    assert levels == ["Low", "Medium", "High"]


def test_saved_model_probabilities_are_valid():
    metadata = json.loads(
        (
            PROJECT_ROOT
            / "models"
            / "model_metadata.json"
        ).read_text(encoding="utf-8")
    )

    model = joblib.load(
        PROJECT_ROOT
        / "models"
        / "churn_risk_pipeline.joblib"
    )

    dataset = pd.read_csv(
        PROJECT_ROOT
        / "data"
        / "processed"
        / "cleaned_churn_data.csv"
    )

    cleaned, _ = clean_telco_data(
        dataset,
        require_target=True
    )

    cleaned = (
        cleaned
        .drop_duplicates(
            subset=["customerID"],
            keep="first"
        )
        .reset_index(drop=True)
    )

    engineered = engineer_features(cleaned)

    sample = engineered.head(10)

    probabilities = model.predict_proba(
        sample[metadata["model_features"]]
    )[:, 1]

    assert len(probabilities) == len(sample)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
