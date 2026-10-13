"""PySpark ETL: raw Telco CSV -> cleaned, feature-engineered Parquet + Spark SQL analytics."""
from pyspark.sql import SparkSession, functions as F, Window
from src.config import RAW_CSV, ARTIFACTS

PARQUET_DIR = str(ARTIFACTS / "spark" / "churn_features.parquet")
SERVICE_COLS = ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


def get_spark(name="churn-extension"):
    return (SparkSession.builder.master("local[*]").appName(name)
            .config("spark.sql.shuffle.partitions", "4")
            .config("spark.ui.showConsoleProgress", "false")
            .config("spark.driver.memory", "2g").getOrCreate())


def build_features(spark):
    raw = spark.read.csv(str(RAW_CSV), header=True, inferSchema=True)
    df = (raw
          .withColumn("TotalCharges", F.expr("try_cast(trim(TotalCharges) as double)"))
          .withColumn("label", F.when(F.col("Churn") == "Yes", 1.0).otherwise(0.0)))
    # blank TotalCharges only occurs for tenure == 0 (not billed yet) -> 0 rather than the median
    df = df.withColumn("TotalCharges", F.when(F.col("TotalCharges").isNull() & (F.col("tenure") == 0), 0.0)
                       .otherwise(F.col("TotalCharges")))
    median_tc = df.approxQuantile("TotalCharges", [0.5], 0.001)[0]
    df = df.fillna({"TotalCharges": median_tc})

    df = (df
          .withColumn("avg_monthly_spend", F.col("TotalCharges") / (F.col("tenure") + 1))
          .withColumn("early_churn_risk", (F.col("tenure") < 6).cast("int"))
          .withColumn("tenure_group", F.when(F.col("tenure") <= 12, "New")
                      .when(F.col("tenure") <= 36, "Mid").otherwise("Loyal"))
          .withColumn("price_sensitivity", F.col("MonthlyCharges") / (F.col("avg_monthly_spend") + 1))
          .withColumn("customer_value_score", F.col("tenure") * F.col("MonthlyCharges")))
    df = df.withColumn("service_count",
                       sum(F.when(F.col(c) == "Yes", 1).otherwise(0) for c in SERVICE_COLS))

    # window function: rank customers by monthly charge inside each contract type
    w = Window.partitionBy("Contract").orderBy(F.desc("MonthlyCharges"))
    df = df.withColumn("charge_rank_in_contract", F.percent_rank().over(w))
    return df


def sql_analytics(spark, df):
    df.createOrReplaceTempView("customers")
    return {
        "churn_by_contract": spark.sql("""
            SELECT Contract, COUNT(*) AS customers, ROUND(AVG(label)*100,1) AS churn_pct,
                   ROUND(AVG(MonthlyCharges),2) AS avg_monthly
            FROM customers GROUP BY Contract ORDER BY churn_pct DESC"""),
        "revenue_at_risk": spark.sql("""
            SELECT tenure_group, ROUND(SUM(CASE WHEN label=1 THEN MonthlyCharges END),0) AS churned_mrr,
                   ROUND(SUM(MonthlyCharges),0) AS total_mrr
            FROM customers GROUP BY tenure_group ORDER BY churned_mrr DESC"""),
    }


if __name__ == "__main__":
    spark = get_spark()
    feats = build_features(spark)
    feats.write.mode("overwrite").parquet(PARQUET_DIR)
    print("rows:", feats.count(), "cols:", len(feats.columns), "->", PARQUET_DIR)
    for k, v in sql_analytics(spark, feats).items():
        print(f"\n== {k}"); v.show()
    spark.stop()
