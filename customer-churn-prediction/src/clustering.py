"""Unsupervised customer segmentation (KMeans + GMM comparison) with MLflow tracking."""
import numpy as np, pandas as pd, mlflow
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from src.config import RAW_CSV, FIGURES, ARTIFACTS, MLFLOW_URI, SEED
from src.data_loader import load_data
from src.preprocessing import clean_data, encode_target
from src.feature_engineering import create_features

# Churn is deliberately NOT used for clustering - it is only used afterwards to profile segments.
CLUSTER_FEATURES = ["tenure", "MonthlyCharges", "service_count", "avg_monthly_spend", "SeniorCitizen",
                    "is_month_to_month", "has_fiber", "has_internet", "autopay"]


def _frame():
    raw = load_data(RAW_CSV); ids = raw["customerID"]
    df = create_features(encode_target(clean_data(raw)))
    df["customerID"] = ids.values
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    df["has_fiber"] = (df["InternetService"] == "Fiber optic").astype(int)
    df["has_internet"] = (df["InternetService"] != "No").astype(int)
    df["autopay"] = df["PaymentMethod"].str.contains("automatic").astype(int)
    return df


def choose_k(Xs, ks=range(2, 9)):
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xs)
        rows.append(dict(k=k, inertia=km.inertia_, silhouette=silhouette_score(Xs, km.labels_, sample_size=4000, random_state=SEED),
                         davies_bouldin=davies_bouldin_score(Xs, km.labels_)))
    return pd.DataFrame(rows)


def name_segment(p, overall):
    """Rule-based business label from a segment's profile row."""
    if p["has_internet"] < 0.2: return "Phone-only basics"
    if p["tenure"] > overall["tenure"] * 1.25 and p["service_count"] >= overall["service_count"]: return "Loyal bundlers"
    if p["has_fiber"] > 0.6 and p["is_month_to_month"] > 0.6 and p["MonthlyCharges"] > overall["MonthlyCharges"]: return "High-bill flexible fiber"
    if p["tenure"] < overall["tenure"] * 0.6: return "New & price-conscious"
    return "Mid-tenure standard"


def run(k=None, experiment="churn-segmentation"):
    mlflow.set_tracking_uri(MLFLOW_URI); mlflow.set_experiment(experiment)
    df = _frame(); Xs = StandardScaler().fit_transform(df[CLUSTER_FEATURES])
    sel = choose_k(Xs); sel.to_csv(ARTIFACTS / "kmeans_k_selection.csv", index=False)
    # Silhouette is nearly flat across k (weak structure), so restrict to 4..6 segments a retention team can act on
    # and take the best silhouette inside that band.
    k = k or int(sel[sel.k.between(4, 6)].sort_values("silhouette", ascending=False).iloc[0]["k"])
    with mlflow.start_run(run_name=f"kmeans_k{k}"):
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Xs)
        gmm = GaussianMixture(n_components=k, random_state=SEED, n_init=3).fit(Xs)
        sil_km = silhouette_score(Xs, km.labels_); sil_gmm = silhouette_score(Xs, gmm.predict(Xs))
        mlflow.log_params({"algo": "kmeans", "k": k, "features": ",".join(CLUSTER_FEATURES)})
        mlflow.log_metrics({"silhouette_kmeans": sil_km, "silhouette_gmm": sil_gmm, "inertia": km.inertia_})
        df["segment_id"] = km.labels_
        prof = df.groupby("segment_id")[CLUSTER_FEATURES + ["Churn"]].mean()
        prof["customers"] = df.groupby("segment_id").size()
        overall = df[CLUSTER_FEATURES].mean()
        prof["segment_name"] = [name_segment(r, overall) for _, r in prof.iterrows()]
        # disambiguate duplicate names using relative spend / tenure inside the duplicate group
        for n, grp in prof.groupby("segment_name"):
            if len(grp) > 1:
                hi = grp["MonthlyCharges"].idxmax()
                for i in grp.index:
                    prof.loc[i, "segment_name"] = f"{n} - {'higher spend' if i == hi else 'lower spend'}"
        df["segment_name"] = df["segment_id"].map(prof["segment_name"])
        prof.round(3).to_csv(ARTIFACTS / "segment_profiles.csv"); mlflow.log_artifact(str(ARTIFACTS / "segment_profiles.csv"))
        # figures: PCA scatter + elbow/silhouette
        pcs = PCA(2, random_state=SEED).fit_transform(Xs)
        fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
        for sid, g in df.groupby("segment_id"):
            ax[0].scatter(pcs[g.index, 0], pcs[g.index, 1], s=6, alpha=.5, label=prof.loc[sid, "segment_name"])
        ax[0].legend(markerscale=3, fontsize=7); ax[0].set_title("Segments in PCA space")
        ax[1].plot(sel.k, sel.inertia, "o-"); ax[1].set_title("Elbow (inertia)"); ax[1].set_xlabel("k")
        ax[2].plot(sel.k, sel.silhouette, "o-g"); ax[2].set_title("Silhouette"); ax[2].set_xlabel("k")
        plt.tight_layout(); fig.savefig(FIGURES / "clustering_segments.png", dpi=130); plt.close(fig)
        mlflow.log_artifact(str(FIGURES / "clustering_segments.png"))
    out = df[["customerID", "segment_id", "segment_name"]]
    out.to_csv(ARTIFACTS / "customer_segments.csv", index=False)
    print(sel.round(3).to_string(index=False)); print(f"\nchosen k={k}  silhouette KMeans={sil_km:.3f} GMM={sil_gmm:.3f}")
    print(prof[["segment_name", "customers", "tenure", "MonthlyCharges", "service_count", "Churn"]].round(2).to_string())
    return out, prof


if __name__ == "__main__":
    run()
