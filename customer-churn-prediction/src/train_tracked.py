"""Train from the Feast feature store, track everything in MLflow, register the champion model.

Experiment grid: 3 models x 3 feature sets (base / +segment / +segment+sentiment). The sentiment features come
from SIMULATED tickets, so that comparison is tagged `synthetic_features=true` and is a pipeline demo, not a claim.
"""
import sys, numpy as np, pandas as pd, mlflow, mlflow.sklearn
from pathlib import Path
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, precision_score, average_precision_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "feature_store" / "feature_repo"))
from feast import FeatureStore
from feature_store.feature_repo import features as defs
from src.config import MLFLOW_URI, RAW_CSV, FIGURES, ARTIFACTS, SEED

REGISTERED_NAME = "churn-risk-model"
SENT_COLS = ["ticket_count", "sentiment_mean", "sentiment_min", "negative_ticket_share"]


def get_training_frame():
    store = FeatureStore(repo_path=str(ROOT / "feature_store" / "feature_repo"))
    raw = pd.read_csv(RAW_CSV)
    entity_df = pd.DataFrame({"customerID": raw.customerID, "event_timestamp": pd.Timestamp("2026-06-02", tz="UTC"),
                              "churn": (raw.Churn == "Yes").astype(int)})
    feats = defs.PROFILE_FEATURES + defs.BEHAVIOR_FEATURES + defs.ENGAGEMENT_FEATURES
    return store.get_historical_features(entity_df=entity_df, features=feats).to_df()


def feature_sets(df):
    cat_base = [c for c in defs.profile_cols] + ["tenure_group"]
    num_base = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend", "early_churn_risk",
                "price_sensitivity", "customer_value_score", "service_count"]
    return {"base": (cat_base, num_base),
            "base+segment": (cat_base + ["segment_name"], num_base),
            "base+segment+sentiment": (cat_base + ["segment_name"], num_base + SENT_COLS)}


def make_model(kind, cat, num):
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
                             ("num", StandardScaler() if kind == "logreg" else "passthrough", num)])
    clf = {"logreg": LogisticRegression(max_iter=1000, class_weight="balanced", C=0.5),
           "random_forest": RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=20,
                                                   class_weight="balanced", random_state=SEED, n_jobs=-1),
           "hist_gbm": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200,
                                                      class_weight="balanced", random_state=SEED)}[kind]
    return Pipeline([("pre", pre), ("clf", clf)])


def shap_plot(pipe, X_sample, path):
    Xt = pipe.named_steps["pre"].transform(X_sample); names = pipe.named_steps["pre"].get_feature_names_out()
    sv = shap.TreeExplainer(pipe.named_steps["clf"]).shap_values(Xt)
    sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if np.ndim(sv) == 3 else sv)   # shap>=0.45 returns [n, f, classes]
    shap.summary_plot(sv, Xt, feature_names=names, show=False, max_display=15); plt.tight_layout()
    plt.savefig(path, dpi=130); plt.close()


def run(experiment="churn-feature-store-training"):
    mlflow.set_tracking_uri(MLFLOW_URI); mlflow.set_experiment(experiment)
    df = get_training_frame(); y = df.pop("churn"); df = df.drop(columns=["customerID", "event_timestamp"])
    tr_idx, te_idx = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=SEED)
    results = []
    for fs_name, (cat, num) in feature_sets(df).items():
        for kind in ["logreg", "random_forest", "hist_gbm"]:
            with mlflow.start_run(run_name=f"{kind}__{fs_name}") as run:
                pipe = make_model(kind, cat, num); Xtr, Xte = df.iloc[tr_idx][cat + num], df.iloc[te_idx][cat + num]
                pipe.fit(Xtr, y.iloc[tr_idx]); p = pipe.predict_proba(Xte)[:, 1]
                cv = cross_val_score(make_model(kind, cat, num), Xtr, y.iloc[tr_idx], scoring="roc_auc",
                                     cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
                m = dict(roc_auc=roc_auc_score(y.iloc[te_idx], p), pr_auc=average_precision_score(y.iloc[te_idx], p),
                         recall_churn=recall_score(y.iloc[te_idx], p > .5), precision_churn=precision_score(y.iloc[te_idx], p > .5),
                         cv_auc_mean=cv.mean(), cv_auc_std=cv.std())
                mlflow.log_params({"model": kind, "feature_set": fs_name, "n_features_raw": len(cat) + len(num),
                                   "feature_source": "feast:churn_features"})
                mlflow.log_metrics(m)
                mlflow.set_tags({"synthetic_features": str("sentiment" in fs_name).lower(), "data_snapshot": "2026-06-01"})
                if kind != "logreg":
                    shap_plot(pipe, Xte.sample(500, random_state=SEED), FIGURES / f"shap_{kind}_{fs_name}.png".replace("+", "_"))
                    mlflow.log_artifact(str(FIGURES / f"shap_{kind}_{fs_name}.png".replace("+", "_")))
                sig = infer_signature(Xte, pipe.predict_proba(Xte)[:, 1])
                mlflow.sklearn.log_model(pipe, name="model", signature=sig, input_example=Xte.head(3),
                                         serialization_format="cloudpickle")  # our own trusted artifacts (MLflow 3 default skops rejects tree types)
                results.append(dict(run_id=run.info.run_id, model=kind, feature_set=fs_name, **m))
    res = pd.DataFrame(results); res.drop(columns="run_id").round(4).to_csv(ARTIFACTS / "experiment_results.csv", index=False)
    print(res.drop(columns="run_id").round(4).to_string(index=False))

    # register champion = best CV AUC among the REAL-feature sets (never promote a model that depends on simulated text)
    real = res[~res.feature_set.str.contains("sentiment")].sort_values("cv_auc_mean", ascending=False).iloc[0]
    mv = mlflow.register_model(f"runs:/{real.run_id}/model", REGISTERED_NAME)
    MlflowClient().set_registered_model_alias(REGISTERED_NAME, "champion", mv.version)
    print(f"\nchampion -> {REGISTERED_NAME} v{mv.version}: {real.model} / {real.feature_set} (CV AUC {real.cv_auc_mean:.4f}, test AUC {real.roc_auc:.4f})")
    return res


if __name__ == "__main__":
    run()
