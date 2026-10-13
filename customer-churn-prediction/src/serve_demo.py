"""Online scoring: Feast online store (features) + MLflow Model Registry (@champion alias) -> churn risk for one customer."""
import sys, pandas as pd, mlflow
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "feature_store" / "feature_repo"))
from feast import FeatureStore
from feature_store.feature_repo import features as defs
from src.config import MLFLOW_URI
from src.train_tracked import REGISTERED_NAME, feature_sets

_store = _model = None


def score_customer(customer_id: str) -> dict:
    global _store, _model
    mlflow.set_tracking_uri(MLFLOW_URI)
    _store = _store or FeatureStore(repo_path=str(ROOT / "feature_store" / "feature_repo"))
    _model = _model or mlflow.sklearn.load_model(f"models:/{REGISTERED_NAME}@champion")
    feats = defs.PROFILE_FEATURES + defs.BEHAVIOR_FEATURES + defs.ENGAGEMENT_FEATURES
    row = _store.get_online_features(features=feats, entity_rows=[{"customerID": customer_id}]).to_df()
    cat, num = feature_sets(row)["base+segment"]
    p = float(_model.predict_proba(row[cat + num])[:, 1][0])
    tier = "High" if p > .7 else "Medium" if p > .4 else "Low"
    return {"customerID": customer_id, "churn_probability": round(p, 3), "risk_tier": tier, "segment": row.segment_name[0]}


if __name__ == "__main__":
    for cid in ["7590-VHVEG", "5575-GNVDE", "9237-HQITU"]: print(score_customer(cid))
