import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate business-driven features.
    """
    df = df.copy()

    # Time-based behavior proxies
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["early_churn_risk"] = (df["tenure"] < 6).astype(int)

    # Tenure segmentation
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 36, 72],
        labels=["New", "Mid", "Loyal"]
    )

    # CRITICAL FIX: ensure it gets one-hot encoded
    df["tenure_group"] = df["tenure_group"].astype(str)

    return df
