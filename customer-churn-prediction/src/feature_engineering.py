import pandas as pd

SERVICE_COLS = [
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate business-driven features (same logic as notebooks/feature_engineering.ipynb)."""
    df = df.copy()

    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["early_churn_risk"] = (df["tenure"] < 6).astype(int)

    # bins start at -1 so tenure == 0 is not turned into NaN
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[-1, 12, 36, 72], labels=["New", "Mid", "Loyal"]
    ).astype(str)

    df["price_sensitivity"] = df["MonthlyCharges"] / (df["avg_monthly_spend"] + 1)
    df["customer_value_score"] = df["tenure"] * df["MonthlyCharges"]
    df["service_count"] = df[SERVICE_COLS].apply(lambda x: (x == "Yes").sum(), axis=1)
    return df
