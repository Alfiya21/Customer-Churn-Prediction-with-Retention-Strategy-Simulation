import pandas as pd


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load churn dataset from CSV file.
    """
    df = pd.read_csv(filepath)
    return df
