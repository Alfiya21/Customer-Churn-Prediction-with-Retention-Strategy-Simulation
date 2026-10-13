# Customer Churn Prediction with Retention Strategy Simulation

An end-to-end, explainable machine learning system for a telecom company. It **predicts which customers will churn, explains why, segments customers, recommends retention offers, and estimates the business impact** of acting on those predictions.

The project has two layers:

1. **Core pipeline** – data cleaning, feature engineering, a cost-sensitive churn model, SHAP explainability and a retention-strategy simulation.
2. **Production-style extension** – PySpark, MLflow, a Feast feature store, customer clustering, a next-best-offer recommender, sentiment features, forecasting and EDA.

> **Dataset:** IBM/Kaggle *Telco Customer Churn* (7,043 customers, 26.5% churn). It is a single snapshot with no text and no calendar, so some extension components use **simulated data**. These are clearly marked below.

---

## Table of contents

- [Key results](#key-results)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Quick start](#quick-start)
- [Core pipeline](#core-pipeline)
- [Extension modules](#extension-modules)
- [Experiment tracking and model registry](#experiment-tracking-and-model-registry)
- [Feature store](#feature-store)
- [Retention strategy simulation](#retention-strategy-simulation)
- [Limitations and honest caveats](#limitations-and-honest-caveats)
- [Fixes made to the original code](#fixes-made-to-the-original-code)
- [Future work](#future-work)
- [Author](#author)

---

## Key results

All numbers are from the committed code on a stratified 80/20 split (seed 42) unless stated otherwise.

| Area | Result |
|---|---|
| Churn model (sklearn Random Forest, cost-sensitive) | ROC-AUC **0.846**, churn recall **0.79** at the 0.5 threshold |
| Model comparison (9 MLflow runs) | Logistic Regression, Random Forest and HistGradientBoosting all land at about **0.84 AUC** (5-fold CV 0.843 to 0.849). Differences are within noise. |
| Spark MLlib | LogReg 0.845, RF 0.841, GBT 0.834, CV-tuned RF 0.838 (same ballpark as sklearn) |
| Customer segments | 6 segments (KMeans), churn rate from **5%** (loyal bundlers) to **54%** (high-bill month-to-month fiber) |
| Next-best add-on recommender | Hit-rate@1 on 783 held-out customers: feature-aware LogReg **0.54**, item-item CF 0.49, popularity 0.47, Spark ALS 0.38 |
| Revenue at risk | About **150K** expected MRR lost over 6 months with no action (assumption-driven scenario) |
| Segment and sentiment features | **No AUC gain** (0.843 to 0.845 with or without). Segments derive from the same inputs and the text is simulated. |

Top churn drivers (from EDA tests and SHAP): month-to-month contracts (Cramér's V 0.41), no tech support (0.34), fiber-optic internet (0.32), electronic-check payment (0.30), and low tenure. Gender has no relationship with churn (p = 0.49).

---

## Architecture

```
                 Raw Telco CSV
                       |
        +--------------+---------------+
        |                              |
  pandas pipeline                PySpark ETL + Spark SQL
  (cleaning, features)           (Parquet, window functions)
        |                              |
        +--------------+---------------+
                       |
     +-----------------+------------------+----------------------+
     |                 |                  |                      |
 Clustering        Sentiment          Recommender           Forecasting
 (KMeans/GMM)   (VADER, simulated)  (CF / ALS / LogReg)   (ETS / SARIMA)
     |                 |                  |                      |
     +--------+--------+                  |                      |
              |                           |                      |
      Feast feature store                 |                      |
   (offline Parquet + online SQLite)      |                      |
              |                           |                      |
   Model training (sklearn, MLflow)       |                      |
   Model Registry  (@champion)            |                      |
              |                           |                      |
              +-------------+-------------+----------------------+
                            |
              Per-customer retention offers
              (risk + segment + next-best add-on)
```

---

## Tech stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Data and ML | pandas, NumPy, scikit-learn, SciPy, statsmodels |
| Big data | PySpark (Spark SQL, MLlib, ALS) |
| MLOps | MLflow (tracking, model registry), Feast (feature store) |
| Explainability | SHAP |
| NLP | VADER sentiment |
| Visualization | matplotlib, seaborn |

---

## Project structure

```
customer-churn-prediction/
├── data/                         # raw Telco CSV + feature-engineered outputs
├── notebooks/                    # EDA and feature-engineering notebooks
├── src/
│   ├── config.py                 # paths, MLflow URI, seed
│   ├── data_loader.py            # load raw data
│   ├── preprocessing.py          # cleaning and target encoding
│   ├── feature_engineering.py    # business-driven features
│   ├── train_model.py            # cost-sensitive Random Forest
│   ├── explainability.py         # SHAP summary plots
│   ├── eda.py                    # data-quality checks + statistical tests
│   ├── spark_etl.py              # PySpark ETL + Spark SQL analytics
│   ├── spark_ml.py               # Spark MLlib models + CV tuning
│   ├── clustering.py             # customer segmentation
│   ├── recommender.py            # next-best add-on recommender
│   ├── text_sentiment.py         # simulated tickets + VADER features
│   ├── scoring.py                # out-of-fold churn probabilities
│   ├── forecasting.py            # revenue-at-risk + time-series backtests
│   ├── train_tracked.py          # feature-store training + MLflow registry
│   ├── serve_demo.py             # online scoring demo
│   └── retention_offers.py       # per-customer action table
├── feature_store/
│   ├── build_feature_store.py    # builds sources, applies registry, materializes
│   └── feature_repo/             # Feast definitions (features.py, feature_store.yaml)
├── reports/figures/              # EDA, clustering, forecasting, SHAP plots
├── artifacts/                    # generated outputs (git-ignored)
├── run_pipeline.py               # original core pipeline
├── run_extended_pipeline.py      # full extended pipeline (9 steps)
├── EXTENSION.md                  # short extension summary
└── requirements.txt
```

---

## Quick start

### 1. Prerequisites

- Python 3.10 or newer
- **Java 17 or newer** (required by PySpark). Check with `java -version`.

### 2. Install

```bash
git clone https://github.com/Alfiya21/Customer-Churn-Prediction-with-Retention-Strategy-Simulation.git
cd Customer-Churn-Prediction-with-Retention-Strategy-Simulation/customer-churn-prediction

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run

```bash
# Original core pipeline
python run_pipeline.py

# Full extended pipeline (9 steps)
python run_extended_pipeline.py

# Faster run: skip the slowest step (Spark cross-validation)
python run_extended_pipeline.py --skip-spark-ml
```

### 4. Explore results

```bash
mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db
# open http://127.0.0.1:5000
```

```bash
# Score a single customer from the online feature store + registered model
python -m src.serve_demo
```

### Pipeline steps

| # | Step | Output |
|---|---|---|
| 1 | PySpark ETL | `artifacts/spark/churn_features.parquet` |
| 2 | Clustering | `artifacts/customer_segments.csv` |
| 3 | Sentiment (simulated) | `artifacts/customer_sentiment_features.csv` |
| 4 | Feast feature store | offline and online stores |
| 5 | Train from feature store | MLflow runs + registered `churn-risk-model` |
| 6 | Spark MLlib models | MLflow experiment `churn-spark-ml` |
| 7 | Recommender | `artifacts/recommender_eval.csv` |
| 8 | Forecasting | `artifacts/revenue_at_risk_projection.csv` |
| 9 | Retention offers | `artifacts/retention_offers.csv` |

---

## Core pipeline

1. **Cleaning** – numeric conversion of `TotalCharges`; blank values occur only for `tenure = 0` (customers not yet billed) and are set to 0.
2. **Feature engineering** (`src/feature_engineering.py`)
   - `avg_monthly_spend` – total charges divided by tenure + 1
   - `early_churn_risk` – flag for tenure under 6 months
   - `tenure_group` – New (0-12), Mid (13-36), Loyal (37-72)
   - `price_sensitivity` – monthly charge relative to historical average spend
   - `customer_value_score` – tenure × monthly charges
   - `service_count` – number of subscribed services
3. **Modeling** – Random Forest with balanced class weights (cost-sensitive learning), because missing a churner costs more than a false alarm.
4. **Explainability** – SHAP summary plots to show which features push a customer toward churn.

---

## Extension modules

### PySpark and Spark MLlib
`spark_etl.py` reproduces the feature engineering in PySpark, adds window functions (percent rank of charges within contract type) and Spark SQL analytics (churn by contract, revenue at risk by tenure group), and writes Parquet. `spark_ml.py` builds MLlib pipelines (StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler) with class weights, trains LogReg, RF and GBT, and tunes RF with `CrossValidator`.

### Customer segmentation
KMeans compared with Gaussian Mixture, with k chosen using elbow, silhouette and Davies-Bouldin scores. Silhouette is nearly flat across k (0.27 to 0.35), meaning cluster structure is weak, so k is restricted to 4 to 6 for actionability and chosen by best silhouette inside that band. Churn is **not** used for clustering; it is only used afterwards to profile segments.

| Segment | Customers | Churn rate |
|---|---|---|
| Loyal bundlers - lower spend | 1,134 | ~5% |
| Phone-only basics | 1,527 | ~7% |
| Loyal bundlers - higher spend | 1,007 | ~16% |
| New and price-conscious | 1,209 | ~32% |
| High-bill flexible fiber - lower spend | 751 | ~52% |
| High-bill flexible fiber - higher spend | 1,415 | ~54% |

### Next-best add-on recommender
Built on the **real** add-on subscription matrix (online security, backup, device protection, tech support, streaming TV and movies) for customers with internet service. Evaluated by leave-one-out hit-rate@k: hide one owned add-on and check where the model ranks it.

| Model | Hit@1 | Hit@2 | Hit@3 |
|---|---|---|---|
| Popularity baseline | 0.470 | 0.702 | 0.867 |
| Item-item collaborative filtering | 0.488 | 0.738 | 0.876 |
| **Feature-aware logistic regression** | **0.544** | **0.792** | **0.922** |
| Spark ALS (implicit feedback) | 0.379 | 0.695 | 0.872 |

ALS underperforms popularity because there are only six items; matrix factorization needs a larger catalog.

### Sentiment (simulated)
`text_sentiment.py` generates support-ticket text from customer attributes (**never from the churn label**), scores it with VADER, and aggregates customer-level features: ticket count, mean and minimum sentiment, negative-ticket share. Churners end up with more and more negative tickets (about 1.5 vs 0.9 tickets, mean sentiment -0.06 vs +0.04) because the generator reflects known churn drivers. Replace `generate_tickets` with a real ticket export for production use.

### Forecasting
- **Part A (model-driven):** expected MRR lost over 6 months, with and without the retention campaign, using out-of-fold churn probabilities.
- **Part B (simulated panel):** seasonal-naive vs Holt-Winters ETS vs SARIMA with rolling-origin backtests on a simulated 42-month panel. These accuracy numbers demonstrate methodology only.

### EDA and data quality
Duplicate and missing-value checks, chi-square tests with Cramér's V, Mann-Whitney U tests, and an overview figure at `reports/figures/eda_overview.png`.

---

## Experiment tracking and model registry

Six MLflow experiments are tracked in a local SQLite backend:

| Experiment | Contents |
|---|---|
| `churn-feature-store-training` | 3 models × 3 feature sets, metrics, SHAP plots, logged models with signatures |
| `churn-spark-ml` | Spark MLlib models and CV result |
| `churn-segmentation` | k, silhouette scores, segment profiles, figures |
| `churn-recommender` | hit-rate@k per model |
| `churn-forecasting` | backtest errors, campaign net gain |

The best model on **real-feature** sets (by 5-fold CV AUC) is registered as `churn-risk-model` with the alias `@champion`. A model that depends on simulated text is never promoted.

---

## Feature store

Feast project `churn_features` with one entity (`customer`, key `customerID`) and three feature views:

| Feature view | Contents |
|---|---|
| `customer_profile` | contract, services, payment method, demographics |
| `customer_behavior` | tenure, charges, engineered features |
| `customer_engagement` | segment, sentiment features |

- **Offline store:** Parquet files, used for point-in-time-correct training data via `get_historical_features`.
- **Online store:** SQLite, used for low-latency lookups via `get_online_features`.
- Training and serving read from the same definitions, which prevents training/serving skew.

CLI usage inside `feature_store/feature_repo/`:

```bash
feast apply
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

---

## Retention strategy simulation

Customers are tiered by out-of-fold churn probability: **High** (> 0.7), **Medium** (0.4 to 0.7), **Low** (< 0.4). Each tier gets an action:

- Segment playbook (for example, a 12-month contract offer with a bill credit for high-bill fiber customers)
- Next-best add-on, ranked by *adoption probability × observed churn-reduction association*

Campaign economics use **assumed** retention rates (High 30%, Medium 20%, Low 5%) and **assumed** offer costs, with the offer effect lasting 3 months.

| Risk tier | Customers | MRR saved (6 mo) | Break-even cost per offer |
|---|---|---|---|
| High | 1,517 | ~6.9K | ~4.6 |
| Medium | 1,887 | ~4.5K | ~2.4 |
| Low | 3,639 | ~0.7K | ~0.2 |

Under the assumed costs the broad campaign does not pay back within 6 months. The break-even column shows how cheap an offer must be for it to pay off. Change the assumptions in `src/forecasting.py`.

---

## Limitations and honest caveats

1. **Simulated data.** The Telco file has no text, no calendar, and contains survivors only. The ticket text and the monthly time-series panel are simulated. Treat them as pipelines, not findings.
2. **Assumption-driven economics.** Retention rates, offer costs and the monthly hazard scaling are assumptions, not measured values.
3. **Correlation, not causation.** Customers with tech support or online security churn less, but that does not prove selling those add-ons will reduce churn. An A/B test would be needed.
4. **Model gains are small.** Several models reach about 0.84 AUC; choosing between them is within noise. Segment and sentiment features add no measurable lift.
5. **Weak cluster structure.** Silhouette around 0.34 means segments are useful for business storytelling but not sharply separated.
6. **Single snapshot.** There is no temporal validation; the 80/20 split is random.

---

## Fixes made to the original code

- The data folder was named `" data"` (leading space), breaking `run_pipeline.py`; renamed to `data`.
- `tenure_group` produced NaN for tenure 0; bins now start at -1.
- Blank `TotalCharges` (new customers) was filled with the median, creating a fake spending outlier; it is now 0.
- SHAP indexing now handles both old and new SHAP return shapes.

---

## Future work

- Replace simulated tickets with a real complaint dataset and fine-tune a transformer sentiment model.
- Add survival analysis (Cox / Kaplan-Meier) for time-to-churn.
- Add uplift modeling to estimate the causal effect of offers.
- Serve the champion model behind a FastAPI endpoint with Docker.
- Add data-drift monitoring and scheduled retraining.
- Add unit tests and CI (GitHub Actions).

---

## Author

**Alfiya Mulla**
B.E. Computer Science & Engineering (Data Science), D.Y. Patil College of Engineering & Technology, Kolhapur

- Portfolio: [alfiyamulla.netlify.app](https://alfiyamulla.netlify.app)
- GitHub: [github.com/Alfiya21](https://github.com/Alfiya21)
- LinkedIn: [linkedin.com/in/alfiyamulla](https://www.linkedin.com/in/alfiyamulla)

## License

Add a license of your choice (MIT is common for portfolio projects) before publishing.
