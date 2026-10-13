"""Support-ticket sentiment pipeline.

IMPORTANT: the Telco dataset has no text. `generate_tickets` SIMULATES ticket text from customer
attributes (not from the churn label) so that the pipeline - text -> VADER sentiment -> customer-level
features -> model - is fully runnable. Swap `generate_tickets` for a real ticket export in production.
"""
import numpy as np, pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.config import RAW_CSV, ARTIFACTS, SEED

POS = ["Thanks, the technician fixed my {t} issue quickly. Great service!",
       "Very happy with the {t}. Support was helpful and friendly.",
       "Excellent experience, my {t} works perfectly now."]
NEU = ["I would like to know how to change my plan regarding {t}.",
       "Can you confirm the date of my next bill and the {t} charges?",
       "Please update my address for {t} service."]
NEG = ["My {t} has been terrible for days and nobody is helping. Very frustrated!",
       "This bill is way too high for the poor {t} I get. I am thinking of cancelling.",
       "Awful support. The {t} keeps failing and I am extremely disappointed.",
       "I was overcharged again. This is unacceptable and I want a refund for the {t}."]
TOPICS = {"billing": "billing", "internet": "internet speed", "outage": "connection outage",
          "streaming": "streaming service", "support": "customer support"}


def generate_tickets(seed=SEED):
    rng = np.random.default_rng(seed); raw = pd.read_csv(RAW_CSV); rows = []
    for r in raw.itertuples():
        # latent dissatisfaction driven by observable attributes + noise (NO use of the churn label)
        d = (0.9 * (r.Contract == "Month-to-month") + 0.7 * (r.InternetService == "Fiber optic")
             + 0.5 * (r.TechSupport == "No") + 0.5 * (r.PaymentMethod == "Electronic check")
             + 0.012 * (r.MonthlyCharges - 65) - 0.015 * r.tenure + rng.normal(0, 0.9))
        n = rng.poisson(max(0.15, 0.6 + 0.5 * d))
        p_neg = 1 / (1 + np.exp(-(d - 0.6))); p_pos = max(0.05, 0.45 - 0.25 * p_neg)
        for _ in range(min(n, 6)):
            u = rng.random(); topic = rng.choice(list(TOPICS))
            pool = NEG if u < p_neg * (1 - p_pos) else (POS if u > 1 - p_pos else NEU)
            rows.append((r.customerID, topic, rng.choice(pool).format(t=TOPICS[topic])))
    return pd.DataFrame(rows, columns=["customerID", "topic", "ticket_text"])


def score_and_aggregate(tickets, all_ids):
    sia = SentimentIntensityAnalyzer()
    tickets = tickets.copy()
    tickets["compound"] = tickets.ticket_text.map(lambda t: sia.polarity_scores(t)["compound"])
    tickets["is_negative"] = (tickets.compound <= -0.05).astype(int)
    agg = tickets.groupby("customerID").agg(ticket_count=("compound", "size"), sentiment_mean=("compound", "mean"),
                                           sentiment_min=("compound", "min"), negative_ticket_share=("is_negative", "mean")).reset_index()
    agg = pd.DataFrame({"customerID": all_ids}).merge(agg, how="left", on="customerID")
    agg["ticket_count"] = agg.ticket_count.fillna(0).astype(int)
    for c in ["sentiment_mean", "sentiment_min"]: agg[c] = agg[c].fillna(0.0)      # no tickets = neutral
    agg["negative_ticket_share"] = agg.negative_ticket_share.fillna(0.0)
    return tickets, agg


def run():
    raw = pd.read_csv(RAW_CSV); tickets = generate_tickets()
    tickets, agg = score_and_aggregate(tickets, raw.customerID)
    tickets.to_csv(ARTIFACTS / "synthetic_tickets_scored.csv", index=False)
    agg.to_csv(ARTIFACTS / "customer_sentiment_features.csv", index=False)
    chk = agg.merge(raw[["customerID", "Churn"]], on="customerID"); chk["churn"] = (chk.Churn == "Yes").astype(int)
    print("tickets:", len(tickets), "| customers with >=1 ticket:", int((agg.ticket_count > 0).sum()))
    print(chk.groupby("churn")[["ticket_count", "sentiment_mean", "negative_ticket_share"]].mean().round(3))
    return tickets, agg


if __name__ == "__main__":
    run()
