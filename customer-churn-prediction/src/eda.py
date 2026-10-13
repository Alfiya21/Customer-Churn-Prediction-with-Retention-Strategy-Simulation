"""EDA + data-quality report -> reports/figures/eda_*.png and artifacts/data_quality_report.csv"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from src.config import RAW_CSV, FIGURES, ARTIFACTS
from src.preprocessing import clean_data, encode_target
from src.feature_engineering import create_features


def data_quality(raw):
    rows = [("rows", len(raw)), ("duplicate_customerIDs", int(raw.customerID.duplicated().sum())),
            ("blank_TotalCharges", int((raw.TotalCharges.astype(str).str.strip() == "").sum())),
            ("blank_TotalCharges_all_have_tenure_0", bool(raw[raw.TotalCharges.astype(str).str.strip() == ""].tenure.eq(0).all())),
            ("churn_rate_pct", round((raw.Churn == "Yes").mean() * 100, 2)),
            ("tenure_range", f"{raw.tenure.min()}-{raw.tenure.max()}")]
    pd.DataFrame(rows, columns=["check", "value"]).to_csv(ARTIFACTS / "data_quality_report.csv", index=False); return rows


def run():
    raw = pd.read_csv(RAW_CSV); dq = data_quality(raw)
    df = create_features(encode_target(clean_data(raw))); sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    sns.barplot(data=df, x="Contract", y="Churn", ax=ax[0, 0], errorbar=None); ax[0, 0].set_title("Churn rate by contract")
    sns.barplot(data=df, x="tenure_group", y="Churn", order=["New", "Mid", "Loyal"], ax=ax[0, 1], errorbar=None); ax[0, 1].set_title("Churn rate by tenure group")
    sns.kdeplot(data=df, x="MonthlyCharges", hue="Churn", fill=True, common_norm=False, ax=ax[0, 2]); ax[0, 2].set_title("Monthly charges by churn")
    sns.barplot(data=df, x="InternetService", y="Churn", ax=ax[1, 0], errorbar=None); ax[1, 0].set_title("Churn by internet service")
    sns.barplot(data=df, x="service_count", y="Churn", ax=ax[1, 1], errorbar=None); ax[1, 1].set_title("Churn by number of services")
    num = df[["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend", "price_sensitivity", "customer_value_score", "service_count", "Churn"]]
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax[1, 2], cbar=False, annot_kws={"size": 7}); ax[1, 2].set_title("Correlation")
    plt.tight_layout(); fig.savefig(FIGURES / "eda_overview.png", dpi=130); plt.close(fig)
    # statistical tests (not just pictures)
    tests = []
    for c in ["Contract", "InternetService", "PaymentMethod", "TechSupport", "gender"]:
        chi2, p, _, _ = stats.chi2_contingency(pd.crosstab(df[c], df.Churn)); n = len(df)
        v = np.sqrt(chi2 / (n * (min(pd.crosstab(df[c], df.Churn).shape) - 1))); tests.append((c, "chi-square", round(p, 4), f"Cramer's V={v:.3f}"))
    for c in ["tenure", "MonthlyCharges", "avg_monthly_spend"]:
        u, p = stats.mannwhitneyu(df[df.Churn == 1][c], df[df.Churn == 0][c]); tests.append((c, "Mann-Whitney U", round(p, 4), ""))
    t = pd.DataFrame(tests, columns=["feature", "test", "p_value", "effect"]); t.to_csv(ARTIFACTS / "eda_stat_tests.csv", index=False)
    print(dq); print(t.to_string(index=False))


if __name__ == "__main__":
    run()
