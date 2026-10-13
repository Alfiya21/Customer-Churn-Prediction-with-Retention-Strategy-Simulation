"""Combine risk (churn model) + segment (clustering) + next-best add-on (recommender) into per-customer actions."""
import numpy as np, pandas as pd
from src.config import ARTIFACTS, RAW_CSV
from src.scoring import oof_scores
from src.recommender import load_matrix, score_feature_lr, recommend_for_all, ITEMS

SEGMENT_PLAYBOOK = {
    "High-bill flexible fiber": "Offer 12-month contract with a bill credit; proactive speed/outage check",
    "New & price-conscious": "Onboarding check-in at day 30/60; starter bundle discount",
    "Loyal bundlers": "Loyalty perk / early renewal reward; avoid heavy discounting",
    "Phone-only basics": "Low-touch email; cross-sell internet only when engaged",
}


def playbook(seg):
    for k, v in SEGMENT_PLAYBOOK.items():
        if str(seg).startswith(k): return v
    return "Standard engagement"


def build():
    raw, R, P = load_matrix(); lr = score_feature_lr(R, P)
    inet0 = raw.assign(churn=(raw.Churn == "Yes").astype(int))
    assoc0 = np.array([inet0[inet0[i] == "Yes"].churn.mean() - inet0[inet0[i] == "No"].churn.mean() for i in ITEMS])
    S = np.where(R == 1, -np.inf, lr(R, np.arange(len(R)), P=P))          # P(customer would adopt each un-owned add-on)
    value = S * np.maximum(-assoc0, 0)                                      # adoption prob x observed churn-reduction association
    best = np.argmax(value, axis=1)
    rec = pd.DataFrame({"customerID": raw.customerID, "rec_1": [ITEMS[b] if np.isfinite(value[i, b]) and value[i, b] > 0 else None for i, b in enumerate(best)],
                        "rec_1_adopt_prob": [S[i, b] if np.isfinite(S[i, b]) else np.nan for i, b in enumerate(best)]})
    scores = oof_scores(); seg = pd.read_csv(ARTIFACTS / "customer_segments.csv")
    out = scores.merge(seg[["customerID", "segment_name"]], on="customerID").merge(rec, on="customerID", how="left")
    out["segment_action"] = out.segment_name.map(playbook)
    # observational association only (NOT causal): churn rate of customers WITH vs WITHOUT each add-on, internet users only
    inet = raw.assign(churn=(raw.Churn == "Yes").astype(int))
    assoc = {i: float(inet[inet[i] == "Yes"].churn.mean() - inet[inet[i] == "No"].churn.mean()) for i in ITEMS}
    pd.Series(assoc, name="churn_rate_diff_owners_minus_nonowners").round(3).to_csv(ARTIFACTS / "addon_churn_association.csv")
    mask = out.risk_segment.isin(["High", "Medium"])
    out["offer"] = np.where(mask, np.where(out.rec_1.notna(), "Bundle offer: " + out.rec_1.fillna("") + " free trial + ", "") + out.segment_action,
                            "Routine engagement")
    out.to_csv(ARTIFACTS / "retention_offers.csv", index=False); return out, assoc


if __name__ == "__main__":
    o, a = build(); print(o.risk_segment.value_counts().to_dict()); print("add-on vs churn (diff in churn rate):", {k: round(v, 3) for k, v in a.items()})
    print(o[o.risk_segment == "High"][["customerID", "churn_probability", "segment_name", "rec_1", "offer"]].head(5).to_string(index=False))
