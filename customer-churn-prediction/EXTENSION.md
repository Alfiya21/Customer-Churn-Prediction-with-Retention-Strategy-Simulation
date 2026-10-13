# Extension: Spark, MLflow, Feast, Segmentation, Recommender, Sentiment, Forecasting

Run everything: `pip install -r requirements.txt && python run_extended_pipeline.py` (add `--skip-spark-ml` for a faster run).
Browse experiments: `mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db`

| Skill | Where | What it does |
|---|---|---|
| PySpark / Spark SQL | `src/spark_etl.py` | Cleaning, feature engineering, window functions, SQL analytics -> Parquet |
| Spark MLlib | `src/spark_ml.py` | Pipeline (StringIndexer, OneHot, VectorAssembler, Scaler), LogReg / RF / GBT with class weights, CrossValidator |
| MLflow | all `src/*.py` | 6 experiments, params/metrics/artifacts, signatures, Model Registry with `@champion` alias |
| Feature store | `feature_store/` | Feast: 1 entity, 3 feature views, offline (Parquet) + online (SQLite) store, point-in-time retrieval for training, online lookup for serving |
| Clustering | `src/clustering.py` | KMeans vs GMM, k selection (elbow, silhouette, Davies-Bouldin), named segments with churn profile |
| Recommendation | `src/recommender.py` | Next-best add-on: popularity vs item-item CF vs Spark ALS vs feature-aware LogReg, leave-one-out hit-rate@k |
| Sentiment | `src/text_sentiment.py` | VADER on **simulated** support tickets -> customer-level features |
| Forecasting | `src/forecasting.py` | 6-month revenue-at-risk projection (model-driven) + ETS / SARIMA / seasonal-naive backtests (simulated panel) |
| EDA / processing | `src/eda.py` | Data-quality checks, chi-square / Mann-Whitney tests, effect sizes |
| Serving | `src/serve_demo.py` | Feast online features + registry model -> churn score for one customer |

## Results (this run)
- Churn models, all ~0.84 ROC-AUC: sklearn LogReg 0.845, RF 0.844, HistGBM 0.839; Spark LogReg 0.845, RF 0.841, GBT 0.834, tuned RF 0.838.
- Segment and sentiment features do **not** improve AUC (0.843-0.845 with or without). Segments are derived from the same inputs; the text is simulated.
- Segments (k=6, silhouette 0.34, weak structure): churn ranges from 5% (loyal bundlers) to 54% (high-bill, month-to-month fiber).
- Recommender (783 held-out users, hit-rate@1): feature-aware LogReg 0.54, item-CF 0.49, popularity 0.47, Spark ALS 0.38 (ALS is worse than popularity with only 6 items).
- Revenue at risk: ~150k expected MRR lost over 6 months with no action. With the assumed offer costs, the broad campaign does not pay back in 6 months; break-even cost per offer is ~4.6 (High tier) and ~2.4 (Medium).

## Caveats (read before quoting any number)
1. The Telco data has **no text, no calendar, and survivors only**. Ticket text and the monthly panel are **simulated** (`text_sentiment.py`, `forecasting.py`). Describe them as pipelines/methodology, not as findings.
2. Campaign economics use assumed retention rates and offer costs.
3. Add-on vs churn differences are observational, not causal.
4. Fixes made to the original code: blank `TotalCharges` (tenure 0) now 0 instead of the median; `tenure_group` no longer produces NaN; SHAP indexing works with new SHAP versions; `" data"` folder renamed to `data`.
