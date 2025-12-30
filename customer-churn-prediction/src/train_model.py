import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


def prepare_train_test(df: pd.DataFrame):
    """
    Split dataset and encode categorical features.
    """
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    cat_cols = X.select_dtypes(include="object").columns
    num_cols = X.select_dtypes(exclude="object").columns

    encoder = OneHotEncoder(drop="first", sparse_output=False)


    X_cat = encoder.fit_transform(X[cat_cols])

    X_cat_df = pd.DataFrame(
        X_cat,
        columns=encoder.get_feature_names_out(cat_cols)
    )

    X_final = pd.concat(
        [X[num_cols].reset_index(drop=True),
         X_cat_df.reset_index(drop=True)],
        axis=1
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_final,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    return X_train, X_test, y_train, y_test, encoder


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """
    Train Random Forest with class balancing.
    """
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train, y_train)
    return model
