import pandas as pd
import numpy as np


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values and data types.
    """
    df = df.copy()

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())



    # Drop customer ID
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode target variable Churn.
    """
    df = df.copy()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df
