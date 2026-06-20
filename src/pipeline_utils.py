from pathlib import Path
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn"
]

SCORING_REQUIRED_COLUMNS = [
    column for column in REQUIRED_COLUMNS
    if column != "Churn"
]

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

CANONICAL_COLUMN_MAP = {
    column.lower(): column
    for column in REQUIRED_COLUMNS
}


def canonicalize_columns(df):
    result = df.copy()
    rename_map = {}

    for column in result.columns:
        normalized = str(column).strip().lower()

        if normalized in CANONICAL_COLUMN_MAP:
            rename_map[column] = CANONICAL_COLUMN_MAP[normalized]

    result = result.rename(columns=rename_map)
    return result


def validate_required_columns(df, require_target=True):
    expected = (
        REQUIRED_COLUMNS
        if require_target
        else SCORING_REQUIRED_COLUMNS
    )

    missing = [
        column for column in expected
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "The dataset is missing required columns: "
            + ", ".join(missing)
        )

    return True


def count_blank_strings(df):
    blank_count = 0

    for column in df.select_dtypes(include=["object", "string"]).columns:
        blank_count += (
            df[column]
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

    return int(blank_count)


def data_quality_report(df, require_target=True):
    working = canonicalize_columns(df)

    duplicate_ids = 0
    if "customerID" in working.columns:
        duplicate_ids = int(
            working["customerID"].duplicated(keep=False).sum()
        )

    invalid_churn = 0
    if require_target and "Churn" in working.columns:
        churn_values = (
            working["Churn"]
            .astype(str)
            .str.strip()
            .str.title()
        )
        invalid_churn = int(
            (~churn_values.isin(["Yes", "No"])).sum()
        )

    negative_numeric = 0
    for column in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if column in working.columns:
            converted = pd.to_numeric(
                working[column],
                errors="coerce"
            )
            negative_numeric += int((converted < 0).sum())

    metrics = [
        ("Rows", int(len(working))),
        ("Columns", int(len(working.columns))),
        ("Exact duplicate rows", int(working.duplicated().sum())),
        ("Duplicate customer ID records", duplicate_ids),
        ("Missing values", int(working.isna().sum().sum())),
        ("Blank strings", count_blank_strings(working)),
        ("Invalid Churn values", invalid_churn),
        ("Negative numeric values", negative_numeric)
    ]

    return pd.DataFrame(
        metrics,
        columns=["Quality Check", "Count"]
    )


def clean_telco_data(df, require_target=True):
    working = canonicalize_columns(df)
    validate_required_columns(
        working,
        require_target=require_target
    )

    rows_before = len(working)

    # Standardize text values and expose blank strings as missing.
    for column in working.select_dtypes(
        include=["object", "string"]
    ).columns:
        working[column] = (
            working[column]
            .astype(str)
            .str.strip()
            .replace(
                {
                    "": np.nan,
                    "nan": np.nan,
                    "None": np.nan
                }
            )
        )

    # Convert key numeric fields.
    numeric_columns = [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    for column in numeric_columns:
        if column in working.columns:
            working[column] = pd.to_numeric(
                working[column],
                errors="coerce"
            )

    # The public Telco dataset contains blank TotalCharges values for
    # some new customers. Zero is used only when tenure is zero.
    if "TotalCharges" in working.columns:
        zero_tenure_mask = (
            working["tenure"]
            .fillna(0)
            .eq(0)
        )

        working.loc[
            zero_tenure_mask & working["TotalCharges"].isna(),
            "TotalCharges"
        ] = 0

        nonzero_median = working.loc[
            ~zero_tenure_mask,
            "TotalCharges"
        ].median()

        if pd.isna(nonzero_median):
            nonzero_median = 0

        working["TotalCharges"] = (
            working["TotalCharges"]
            .fillna(nonzero_median)
        )

    if require_target and "Churn" in working.columns:
        working["Churn"] = (
            working["Churn"]
            .astype(str)
            .str.strip()
            .str.title()
        )

        invalid_target = (
            ~working["Churn"].isin(["Yes", "No"])
        )

        if invalid_target.any():
            invalid_values = sorted(
                working.loc[
                    invalid_target,
                    "Churn"
                ].dropna().unique().tolist()
            )

            raise ValueError(
                "Unexpected Churn values were found: "
                + str(invalid_values)
            )

    exact_duplicates_removed = int(
        working.duplicated().sum()
    )

    working = (
        working
        .drop_duplicates()
        .reset_index(drop=True)
    )

    duplicate_customer_records = 0
    if "customerID" in working.columns:
        duplicate_customer_records = int(
            working["customerID"]
            .duplicated(keep=False)
            .sum()
        )

    audit = {
        "Rows before cleaning": int(rows_before),
        "Rows after exact-duplicate removal": int(len(working)),
        "Exact duplicate rows removed": exact_duplicates_removed,
        "Duplicate customer ID records flagged":
            duplicate_customer_records
    }

    return working, audit


def engineer_features(df):
    result = df.copy()

    result["TenureBand"] = pd.cut(
        result["tenure"],
        bins=[-np.inf, 12, 24, 48, np.inf],
        labels=[
            "0-12 months",
            "13-24 months",
            "25-48 months",
            "Above 48 months"
        ]
    ).astype("object")

    available_service_columns = [
        column for column in SERVICE_COLUMNS
        if column in result.columns
    ]

    result["ServiceCount"] = (
        result[available_service_columns]
        .eq("Yes")
        .sum(axis=1)
        .astype(int)
    )

    result["LongTermContract"] = (
        result["Contract"]
        .isin(["One year", "Two year"])
        .astype(int)
    )

    result["AutomaticPayment"] = (
        result["PaymentMethod"]
        .astype(str)
        .str.contains(
            "automatic",
            case=False,
            na=False
        )
        .astype(int)
    )

    result["HasSupport"] = (
        result[["OnlineSecurity", "TechSupport"]]
        .eq("Yes")
        .any(axis=1)
        .astype(int)
    )

    return result


def assign_risk_level(
    probability,
    low_threshold=0.30,
    high_threshold=0.60
):
    probability = float(probability)

    if probability >= high_threshold:
        return "High"

    if probability >= low_threshold:
        return "Medium"

    return "Low"


def observed_risk_factors(
    row,
    monthly_charge_reference=None,
    maximum_factors=3
):
    factors = []

    if str(row.get("Contract", "")) == "Month-to-month":
        factors.append("Month-to-month contract")

    tenure = pd.to_numeric(
        pd.Series([row.get("tenure")]),
        errors="coerce"
    ).iloc[0]

    if pd.notna(tenure) and tenure <= 12:
        factors.append("Short customer tenure")

    if str(row.get("TechSupport", "")) != "Yes":
        factors.append("No technical support")

    if str(row.get("OnlineSecurity", "")) != "Yes":
        factors.append("No online security")

    if "Electronic check" in str(row.get("PaymentMethod", "")):
        factors.append("Electronic check payment")

    monthly_charge = pd.to_numeric(
        pd.Series([row.get("MonthlyCharges")]),
        errors="coerce"
    ).iloc[0]

    if (
        monthly_charge_reference is not None
        and pd.notna(monthly_charge)
        and monthly_charge > monthly_charge_reference
    ):
        factors.append("Above-median monthly charge")

    if str(row.get("InternetService", "")) == "Fiber optic":
        factors.append("Fiber optic service segment")

    if not factors:
        factors.append(
            "No dominant rule-based factor identified"
        )

    return "; ".join(factors[:maximum_factors])


def mask_customer_id(value):
    text = str(value)

    if len(text) <= 4:
        return "****"

    return text[:3] + "***" + text[-2:]
