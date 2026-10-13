"""Next-best add-on recommender built on the REAL add-on subscription matrix in the Telco data.

Customers x 6 add-on services (OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
StreamingTV, StreamingMovies) for the customers that have internet service.
Models compared with leave-one-out hit-rate@k:
  1. Popularity baseline
  2. Item-item collaborative filtering (cosine)
  3. Spark MLlib ALS (implicit feedback)
  4. Feature-aware per-item logistic regression (profile + other owned add-ons)
"""
import numpy as np, pandas as pd, mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.config import RAW_CSV, ARTIFACTS, MLFLOW_URI, SEED

ITEMS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
PROFILE = ["tenure", "MonthlyCharges", "SeniorCitizen", "fiber", "month_to_month", "partner", "dependents"]


def load_matrix():
    raw = pd.read_csv(RAW_CSV)
    raw = raw[raw["InternetService"] != "No"].reset_index(drop=True)
    R = np.column_stack([(raw[c] == "Yes").astype(int) for c in ITEMS])
    prof = pd.DataFrame({"tenure": raw.tenure, "MonthlyCharges": raw.MonthlyCharges, "SeniorCitizen": raw.SeniorCitizen,
                         "fiber": (raw.InternetService == "Fiber optic").astype(int),
                         "month_to_month": (raw.Contract == "Month-to-month").astype(int),
                         "partner": (raw.Partner == "Yes").astype(int), "dependents": (raw.Dependents == "Yes").astype(int)})
    return raw, R, prof


# ---------- scorers: all take (R_hidden, idx) and return score matrix [len(idx), n_items] ----------
def score_popularity(R_train):
    pop = R_train.mean(0)
    return lambda Rh, idx: np.tile(pop, (len(idx), 1))


def score_item_cf(R_train):
    X = R_train.astype(float); norms = np.linalg.norm(X, axis=0) + 1e-9
    sim = (X.T @ X) / np.outer(norms, norms); np.fill_diagonal(sim, 0)
    return lambda Rh, idx: Rh[idx] @ sim


def score_feature_lr(R_train, P_train):
    sc = StandardScaler().fit(P_train); Pt = sc.transform(P_train); models = []
    for i in range(R_train.shape[1]):
        Xi = np.hstack([Pt, np.delete(R_train, i, axis=1)])
        models.append(LogisticRegression(max_iter=500, C=0.5).fit(Xi, R_train[:, i]))

    def f(Rh, idx, P=None):
        Pi = sc.transform(P.iloc[idx]); out = np.zeros((len(idx), Rh.shape[1]))
        for i, m in enumerate(models):
            out[:, i] = m.predict_proba(np.hstack([Pi, np.delete(Rh[idx], i, axis=1)]))[:, 1]
        return out
    return f


def score_als(R_train, train_users, spark=None):
    """Spark ALS with implicit feedback. Returns scorer using learned factors (users folded in via item factors)."""
    from pyspark.ml.recommendation import ALS
    from src.spark_etl import get_spark
    spark = spark or get_spark("als")
    rows = [(int(u), int(i), 1.0) for u, r in zip(train_users, R_train) for i in np.where(r == 1)[0]]
    df = spark.createDataFrame(rows, ["user", "item", "rating"])
    model = ALS(userCol="user", itemCol="item", ratingCol="rating", implicitPrefs=True, rank=4, regParam=0.1,
                alpha=20.0, maxIter=15, seed=SEED, coldStartStrategy="drop").fit(df)
    V = np.array([r.features for r in model.itemFactors.orderBy("id").collect()])  # [items, rank]
    # fold-in new/hidden user vector by ridge regression on item factors: u = argmin ||V u - r||^2 + lam||u||^2
    lam = 0.1; A = np.linalg.inv(V.T @ V + lam * np.eye(V.shape[1])) @ V.T

    def f(Rh, idx):
        U = Rh[idx] @ A.T; return U @ V.T
    return f


def hit_rate(score_fn, R, test_idx, ks=(1, 2, 3), P=None, rng=None):
    """Leave-one-out: hide one owned add-on per test user, rank the non-owned items, check where the hidden one lands."""
    rng = rng or np.random.default_rng(SEED)
    users = [u for u in test_idx if R[u].sum() >= 2]          # need something left to condition on
    hidden = np.array([rng.choice(np.where(R[u] == 1)[0]) for u in users])
    Rh = R.copy(); Rh[users, hidden] = 0
    try: S = score_fn(Rh, np.array(users), P=P)
    except TypeError: S = score_fn(Rh, np.array(users))
    S = np.where(Rh[users] == 1, -np.inf, S)                   # never recommend what is already owned
    ranks = (S > S[np.arange(len(users)), hidden][:, None]).sum(1)   # 0 = top
    return {f"hit_at_{k}": float((ranks < k).mean()) for k in ks} | {"n_eval_users": len(users)}


def run(with_als=True, experiment="churn-recommender"):
    mlflow.set_tracking_uri(MLFLOW_URI); mlflow.set_experiment(experiment)
    raw, R, P = load_matrix()
    rng = np.random.default_rng(SEED); perm = rng.permutation(len(R)); cut = int(.8 * len(R))
    tr, te = perm[:cut], perm[cut:]
    scorers = {"popularity": score_popularity(R[tr]), "item_item_cf": score_item_cf(R[tr]),
               "feature_logreg": score_feature_lr(R[tr], P.iloc[tr])}
    if with_als: scorers["spark_als_implicit"] = score_als(R[tr], tr)
    res = {}
    for name, fn in scorers.items():
        with mlflow.start_run(run_name=name):
            m = hit_rate(fn, R, te, P=P, rng=np.random.default_rng(SEED)); res[name] = m
            mlflow.log_params({"model": name, "items": len(ITEMS)}); mlflow.log_metrics(m)
    out = pd.DataFrame(res).T; out.to_csv(ARTIFACTS / "recommender_eval.csv"); print(out.round(3).to_string())
    return scorers, raw, R, P, out


def recommend_for_all(scorer_lr, raw, R, P, top_n=2):
    S = scorer_lr(R, np.arange(len(R)), P=P)
    S = np.where(R == 1, -np.inf, S)
    top = np.argsort(-S, axis=1)[:, :top_n]
    rec = pd.DataFrame({"customerID": raw.customerID})
    for j in range(top_n):
        rec[f"rec_{j+1}"] = [ITEMS[t] if np.isfinite(S[i, t]) else None for i, t in enumerate(top[:, j])]
        rec[f"rec_{j+1}_prob"] = [S[i, t] if np.isfinite(S[i, t]) else np.nan for i, t in enumerate(top[:, j])]
    return rec


if __name__ == "__main__":
    run()
