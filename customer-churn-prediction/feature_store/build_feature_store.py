"""Build offline parquet sources from the pipeline outputs, `apply` the Feast registry and materialize to the online store."""
import sys, pandas as pd
from datetime import datetime, timezone
from pathlib import Path
HERE = Path(__file__).resolve().parent; REPO = HERE / "feature_repo"
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(REPO))
from feast import FeatureStore
from feature_store.feature_repo import features as defs
from src.config import RAW_CSV, ARTIFACTS
from src.preprocessing import clean_data
from src.feature_engineering import create_features

SNAPSHOT = pd.Timestamp("2026-06-01", tz="UTC")     # assumed snapshot date of the Telco extract


def build_sources():
    raw = pd.read_csv(RAW_CSV); ids = raw.customerID
    df = create_features(clean_data(raw.assign(Churn=raw.Churn)))
    df.insert(0, "customerID", ids.values)
    seg = pd.read_csv(ARTIFACTS / "customer_segments.csv"); sent = pd.read_csv(ARTIFACTS / "customer_sentiment_features.csv")
    ts = {"event_timestamp": SNAPSHOT, "created": pd.Timestamp.now(tz="UTC")}
    prof = df[["customerID", "SeniorCitizen"] + defs.profile_cols].assign(**ts)
    beh = df[["customerID"] + [f.name for f in defs.customer_behavior_fv.schema]].assign(**ts)
    eng = seg.merge(sent, on="customerID").assign(**ts)
    for name, d in [("customer_profile", prof), ("customer_behavior", beh), ("customer_engagement", eng)]:
        d.to_parquet(REPO / "data" / f"{name}.parquet", index=False)
    return len(df)


def main():
    n = build_sources(); store = FeatureStore(repo_path=str(REPO)); store.apply(defs.ALL_OBJECTS)
    store.materialize(start_date=datetime(2020, 1, 1, tzinfo=timezone.utc), end_date=datetime(2026, 7, 1, tzinfo=timezone.utc))
    print(f"{n} customers -> 3 feature views registered + materialized to the online store")
    print("feature views:", [fv.name for fv in store.list_feature_views()])
    demo = store.get_online_features(features=defs.BEHAVIOR_FEATURES + ["customer_engagement:segment_name", "customer_engagement:sentiment_mean"],
                                     entity_rows=[{"customerID": "7590-VHVEG"}]).to_df()
    print("\nonline lookup (7590-VHVEG):"); print(demo.T.to_string(header=False))


if __name__ == "__main__":
    main()
