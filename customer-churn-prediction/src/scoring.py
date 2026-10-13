"""Out-of-fold churn probabilities (so downstream business simulations never use in-sample scores)."""
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from src.config import RAW_CSV, ARTIFACTS, SEED
from src.data_loader import load_data
from src.preprocessing import clean_data, encode_target
from src.feature_engineering import create_features

OOF_PATH = ARTIFACTS / "oof_churn_scores.csv"


def oof_scores(force=False):
    if OOF_PATH.exists() and not force: return pd.read_csv(OOF_PATH)
    raw = load_data(RAW_CSV); ids = raw["customerID"]
    df = create_features(encode_target(clean_data(raw)))
    y = df.pop("Churn"); X = pd.get_dummies(df, drop_first=True).astype(float)
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=20, class_weight="balanced",
                                random_state=SEED, n_jobs=-1)
    p = cross_val_predict(rf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method="predict_proba")[:, 1]
    out = pd.DataFrame({"customerID": ids.values, "churn_probability": p, "actual_churn": y.values,
                        "MonthlyCharges": raw["MonthlyCharges"].values})
    out["risk_segment"] = pd.cut(out.churn_probability, [-1, .4, .7, 2], labels=["Low", "Medium", "High"]).astype(str)
    out.to_csv(OOF_PATH, index=False); return out
