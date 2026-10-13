"""One command for the whole extended project (run from this folder):  python run_extended_pipeline.py

Order matters: each step produces files the next one reads.
  1 PySpark ETL (+Spark SQL)       -> artifacts/spark/churn_features.parquet
  2 Customer segmentation          -> artifacts/customer_segments.csv            (MLflow: churn-segmentation)
  3 Ticket sentiment (simulated)   -> artifacts/customer_sentiment_features.csv
  4 Feast feature store            -> feature_store/feature_repo/data/*.db       (offline + online store)
  5 Train from feature store       -> MLflow runs + Model Registry champion      (MLflow: churn-feature-store-training)
  6 Spark MLlib churn models       -> MLflow: churn-spark-ml
  7 Add-on recommender             -> MLflow: churn-recommender
  8 Forecasting + revenue at risk  -> MLflow: churn-forecasting
  9 Per-customer retention offers  -> artifacts/retention_offers.csv
View experiments:  mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db
"""
import sys, time
from src import spark_etl, clustering, text_sentiment, spark_ml, recommender, forecasting, retention_offers, train_tracked
from feature_store import build_feature_store


def step(n, name, fn):
    t = time.time(); print(f"\n{'=' * 8} [{n}/9] {name} {'=' * 8}"); fn(); print(f"-- done in {time.time() - t:.0f}s")


def etl():
    s = spark_etl.get_spark(); f = spark_etl.build_features(s); f.write.mode("overwrite").parquet(spark_etl.PARQUET_DIR)
    [v.show() for v in spark_etl.sql_analytics(s, f).values()]; s.stop()


if __name__ == "__main__":
    skip_spark_ml = "--skip-spark-ml" in sys.argv          # Spark CV is the slowest step on a laptop
    step(1, "PySpark ETL", etl); step(2, "Clustering", clustering.run); step(3, "Sentiment", text_sentiment.run)
    step(4, "Feature store", build_feature_store.main); step(5, "Train from feature store + register", train_tracked.run)
    if not skip_spark_ml: step(6, "Spark MLlib models", spark_ml.run)
    step(7, "Recommender", recommender.run); step(8, "Forecasting", forecasting.run); step(9, "Retention offers", retention_offers.build)
