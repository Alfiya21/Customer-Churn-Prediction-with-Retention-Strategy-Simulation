"""Spark MLlib churn model (distributed pipeline) logged to MLflow."""
import mlflow
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from src.spark_etl import get_spark, build_features
from src.config import MLFLOW_URI, SEED

CAT = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
       "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
       "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod", "tenure_group"]
NUM = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend",
       "early_churn_risk", "price_sensitivity", "customer_value_score", "service_count"]


def make_pipeline(clf):
    idx = [StringIndexer(inputCol=c, outputCol=f"{c}_i", handleInvalid="keep") for c in CAT]
    ohe = OneHotEncoder(inputCols=[f"{c}_i" for c in CAT], outputCols=[f"{c}_oh" for c in CAT])
    asm = VectorAssembler(inputCols=NUM + [f"{c}_oh" for c in CAT], outputCol="raw_features")
    sc = StandardScaler(inputCol="raw_features", outputCol="features")
    return Pipeline(stages=idx + [ohe, asm, sc, clf])


def run(experiment="churn-spark-ml"):
    mlflow.set_tracking_uri(MLFLOW_URI); mlflow.set_experiment(experiment)
    spark = get_spark("spark-ml")
    df = build_features(spark)
    # class weights (cost-sensitive learning, same idea as class_weight='balanced')
    pos = df.filter("label=1").count(); neg = df.count() - pos
    w_pos = neg / pos
    from pyspark.sql import functions as F
    df = df.withColumn("w", F.when(F.col("label") == 1, w_pos).otherwise(1.0))
    train, test = df.randomSplit([0.8, 0.2], seed=SEED)
    train.cache(); test.cache()
    ev = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    ev_pr = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderPR")

    candidates = {
        "spark_logreg": LogisticRegression(labelCol="label", weightCol="w", maxIter=100),
        "spark_rf": RandomForestClassifier(labelCol="label", weightCol="w", numTrees=100, maxDepth=8, seed=SEED),
        "spark_gbt": GBTClassifier(labelCol="label", weightCol="w", maxIter=30, maxDepth=4, seed=SEED),
    }
    results = {}
    for name, clf in candidates.items():
        with mlflow.start_run(run_name=name):
            model = make_pipeline(clf).fit(train)
            pred = model.transform(test)
            auc, pr = ev.evaluate(pred), ev_pr.evaluate(pred)
            mlflow.log_params({"framework": "spark-ml", "model": name, "class_weight_pos": round(w_pos, 3)})
            mlflow.log_metrics({"roc_auc": auc, "pr_auc": pr})
            results[name] = (auc, pr); print(f"{name:14s} AUC={auc:.4f}  PR-AUC={pr:.4f}")

    # cross-validated tuning of the best-performing family (distributed CV)
    with mlflow.start_run(run_name="spark_rf_cv_tuned"):
        rf = RandomForestClassifier(labelCol="label", weightCol="w", seed=SEED)
        grid = (ParamGridBuilder().addGrid(rf.numTrees, [100])
                .addGrid(rf.maxDepth, [5, 8]).build())
        cv = CrossValidator(estimator=make_pipeline(rf), estimatorParamMaps=grid,
                            evaluator=ev, numFolds=2, parallelism=2, seed=SEED)
        cvm = cv.fit(train)
        auc = ev.evaluate(cvm.transform(test)); best = cvm.bestModel.stages[-1]
        mlflow.log_params({"framework": "spark-ml", "model": "spark_rf_cv", "best_numTrees": best.getNumTrees,
                           "best_maxDepth": best.getOrDefault("maxDepth")})
        mlflow.log_metric("roc_auc", auc); print(f"spark_rf_cv    AUC={auc:.4f}")
    spark.stop(); return results


if __name__ == "__main__":
    run()
