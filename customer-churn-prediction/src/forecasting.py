"""Forecasting, in two honest parts.

PART A - revenue-at-risk projection (REAL model output): expected MRR lost over the next 6 months, with
and without the README's retention campaign, using out-of-fold churn probabilities.

PART B - time-series methods (SIMULATED panel): the Telco file is a single snapshot with survivors only, so
it has no usable calendar. We therefore simulate a 42-month monthly panel (documented assumptions below)
from the real customer base and show seasonal-naive vs Holt-Winters vs SARIMA with rolling-origin backtests.
Part B demonstrates methodology; its accuracy numbers are NOT a claim about real telco revenue.
"""
import warnings, numpy as np, pandas as pd, mlflow
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.config import ARTIFACTS, FIGURES, MLFLOW_URI, SEED, RAW_CSV
from src.scoring import oof_scores

warnings.filterwarnings("ignore")
MONTHLY_SCALE = 0.15          # assumption: label = churn in the observed window; convert to a monthly hazard
RETENTION = {"High": 0.30, "Medium": 0.20, "Low": 0.05}      # from the README's campaign assumptions
OFFER_COST = {"High": 25.0, "Medium": 10.0, "Low": 1.0}      # assumed cost per outreach (INR/USD agnostic)


# ---------------- Part A ----------------
def revenue_at_risk(horizon=6):
    s = oof_scores(); s = s[s.actual_churn == 0].copy() if False else s       # score the whole base
    rows = []
    for scen in ["no_action", "with_campaign"]:
        alive = np.ones(len(s)); lost = []; cost = 0.0
        for m in range(horizon):
            h = s.churn_probability.values * MONTHLY_SCALE
            if scen == "with_campaign" and m == 0:
                r = s.risk_segment.map(RETENTION).values; cost = float(s.risk_segment.map(OFFER_COST).sum())
                campaign_factor = 1 - r
            h_eff = h * (campaign_factor if scen == "with_campaign" and m < 3 else 1.0)   # offer effect lasts 3 months
            lost_m = float((alive * h_eff * s.MonthlyCharges.values).sum()); alive = alive * (1 - h_eff)
            lost.append(lost_m)
        rows.append(dict(scenario=scen, horizon_months=horizon, expected_mrr_lost_per_month=np.round(lost, 0).tolist(),
                         cumulative_mrr_lost=float(np.sum(lost)), campaign_cost=cost if scen == "with_campaign" else 0.0))
    out = pd.DataFrame(rows)
    saved = out.cumulative_mrr_lost[0] - out.cumulative_mrr_lost[1]
    out.attrs["net_gain"] = saved - out.campaign_cost[1]; return out


def tier_economics(horizon=6, effect_months=3):
    """Per risk tier: MRR saved by the offer vs. its cost, plus the break-even cost per offer."""
    s = oof_scores(); rows = []
    for tier, g in s.groupby("risk_segment"):
        h = g.churn_probability.values * MONTHLY_SCALE; c = g.MonthlyCharges.values
        a0 = np.ones(len(g)); a1 = np.ones(len(g)); base = camp = 0.0
        for m in range(horizon):
            h1 = h * ((1 - RETENTION[tier]) if m < effect_months else 1.0)
            base += float((a0 * h * c).sum()); camp += float((a1 * h1 * c).sum()); a0 *= (1 - h); a1 *= (1 - h1)
        saved = base - camp; cost = OFFER_COST[tier] * len(g)
        rows.append(dict(risk_tier=tier, customers=len(g), mrr_saved_6m=round(saved), campaign_cost=round(cost),
                         net=round(saved - cost), breakeven_cost_per_offer=round(saved / len(g), 2)))
    return pd.DataFrame(rows).sort_values("net", ascending=False)


# ---------------- Part B ----------------
def simulate_panel(months=42, seed=SEED):
    rng = np.random.default_rng(seed); raw = pd.read_csv(RAW_CSV); s = oof_scores()
    hazard = s.churn_probability.values * MONTHLY_SCALE; charges = raw.MonthlyCharges.values
    pool_h, pool_c = hazard, charges; n = len(raw)
    alive = np.ones(n, bool); rows = []
    for t in range(months):
        season = 1 + 0.25 * np.sin(2 * np.pi * (t % 12) / 12) + 0.004 * t                     # annual cycle + slow trend
        new_n = rng.poisson(110 * season)
        idx = rng.integers(0, len(pool_c), new_n)
        alive = np.concatenate([alive, np.ones(new_n, bool)]); hazard = np.concatenate([hazard, pool_h[idx]]); charges = np.concatenate([charges, pool_c[idx]])
        h_t = hazard * (1 + 0.20 * np.cos(2 * np.pi * ((t + 2) % 12) / 12))                   # churn seasonality (post-holiday spike)
        churned = alive & (rng.random(len(alive)) < h_t)
        alive &= ~churned
        rows.append(dict(month=pd.Timestamp("2023-01-01") + pd.DateOffset(months=t), mrr=charges[alive].sum(),
                         churned_mrr=charges[churned].sum(), new_mrr=charges[-new_n:].sum() if new_n else 0, active=int(alive.sum())))
    return pd.DataFrame(rows).set_index("month").asfreq("MS")


def seasonal_naive(tr, h, m=12): return np.array([tr.iloc[-m + (i % m)] for i in range(h)])
def ets(tr, h): return ExponentialSmoothing(tr, trend="add", damped_trend=True, seasonal="add", seasonal_periods=12).fit().forecast(h).values
def sarima(tr, h): return SARIMAX(tr, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False).forecast(h).values
MODELS = {"seasonal_naive": seasonal_naive, "holt_winters_ets": ets, "sarima": sarima}
mape = lambda y, f: float(np.mean(np.abs((y - f) / np.maximum(y, 1))) * 100)


def backtest(s, h=6, n_folds=3):
    rows = []
    for f in range(n_folds):
        cut = len(s) - h * (n_folds - f); tr, te = s.iloc[:cut], s.iloc[cut:cut + h]
        for name, fn in MODELS.items(): rows.append(dict(fold=f, model=name, mape=mape(te.values, fn(tr, h))))
    return pd.DataFrame(rows).groupby("model").mape.mean().sort_values()


def run(horizon=6, experiment="churn-forecasting"):
    mlflow.set_tracking_uri(MLFLOW_URI); mlflow.set_experiment(experiment)
    a = revenue_at_risk(horizon); a.to_csv(ARTIFACTS / "revenue_at_risk_projection.csv", index=False)
    print("PART A - 6-month revenue at risk (real model output)"); print(a.to_string(index=False))
    print(f"net gain of campaign over 6 months (assumption-driven): {a.attrs['net_gain']:,.0f}")
    te = tier_economics(); te.to_csv(ARTIFACTS / "retention_tier_economics.csv", index=False)
    print("\nPer-tier campaign economics (assumed costs/retention rates)"); print(te.to_string(index=False))
    panel = simulate_panel(); panel.to_csv(ARTIFACTS / "simulated_monthly_panel.csv")
    res = {}
    for target in ["churned_mrr", "mrr"]:
        bt = backtest(panel[target]); res[target] = bt; print(f"\nPART B - backtest MAPE (%) on SIMULATED {target}"); print(bt.round(2))
    best = res["churned_mrr"].index[0]
    with mlflow.start_run(run_name=f"ts_{best}"):
        mlflow.log_params({"best_model": best, "panel": "simulated", "horizon": horizon})
        mlflow.log_metrics({f"mape_{m}": v for m, v in res["churned_mrr"].items()})
        mlflow.log_metric("net_gain_campaign_6m", a.attrs["net_gain"])
        fc = MODELS[best](panel.churned_mrr, horizon); idx = pd.date_range(panel.index[-1] + pd.DateOffset(months=1), periods=horizon, freq="MS")
        pd.DataFrame({"month": idx, "forecast_churned_mrr": fc, "model": best}).to_csv(ARTIFACTS / "churned_mrr_forecast.csv", index=False)
        fig, ax = plt.subplots(1, 2, figsize=(13, 4))
        panel.churned_mrr.plot(ax=ax[0], label="churned MRR (simulated)"); pd.Series(fc, idx).plot(ax=ax[0], color="crimson", marker="o", label=f"forecast: {best}")
        ax[0].set_title("Forecast of monthly churned MRR (simulated panel)"); ax[0].legend()
        for r in a.itertuples(): ax[1].plot(range(1, horizon + 1), np.cumsum(r.expected_mrr_lost_per_month), marker="o", label=r.scenario)
        ax[1].set_title("Cumulative expected MRR lost (model-driven)"); ax[1].set_xlabel("month ahead"); ax[1].legend()
        plt.tight_layout(); fig.savefig(FIGURES / "forecasting.png", dpi=130); plt.close(fig); mlflow.log_artifact(str(FIGURES / "forecasting.png"))
    return a, res


if __name__ == "__main__":
    run()
