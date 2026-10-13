"""Feast feature definitions: one entity (customer) and three feature views.

Run from this folder with the Feast CLI too:  feast apply && feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
"""
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64, String

customer = Entity(name="customer", join_keys=["customerID"], description="Telco customer")


def _src(name):
    return FileSource(path=f"data/{name}.parquet", timestamp_field="event_timestamp")


profile_cols = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
                "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
                "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod"]

customer_profile_fv = FeatureView(
    name="customer_profile", entities=[customer], ttl=timedelta(days=3650), source=_src("customer_profile"),
    schema=[Field(name="SeniorCitizen", dtype=Int64)] + [Field(name=c, dtype=String) for c in profile_cols],
    tags={"owner": "alfiya", "type": "static-attributes"})

customer_behavior_fv = FeatureView(
    name="customer_behavior", entities=[customer], ttl=timedelta(days=3650), source=_src("customer_behavior"),
    schema=[Field(name="tenure", dtype=Int64), Field(name="MonthlyCharges", dtype=Float64),
            Field(name="TotalCharges", dtype=Float64), Field(name="avg_monthly_spend", dtype=Float64),
            Field(name="early_churn_risk", dtype=Int64), Field(name="tenure_group", dtype=String),
            Field(name="price_sensitivity", dtype=Float64), Field(name="customer_value_score", dtype=Float64),
            Field(name="service_count", dtype=Int64)],
    tags={"owner": "alfiya", "type": "engineered-behaviour"})

customer_engagement_fv = FeatureView(
    name="customer_engagement", entities=[customer], ttl=timedelta(days=3650), source=_src("customer_engagement"),
    schema=[Field(name="segment_id", dtype=Int64), Field(name="segment_name", dtype=String),
            Field(name="ticket_count", dtype=Int64), Field(name="sentiment_mean", dtype=Float64),
            Field(name="sentiment_min", dtype=Float64), Field(name="negative_ticket_share", dtype=Float64)],
    tags={"owner": "alfiya", "type": "segment+sentiment", "note": "sentiment is from SIMULATED tickets"})

ALL_OBJECTS = [customer, customer_profile_fv, customer_behavior_fv, customer_engagement_fv]
PROFILE_FEATURES = [f"customer_profile:{c}" for c in ["SeniorCitizen"] + profile_cols]
BEHAVIOR_FEATURES = [f"customer_behavior:{c.name}" for c in customer_behavior_fv.schema]
ENGAGEMENT_FEATURES = [f"customer_engagement:{c.name}" for c in customer_engagement_fv.schema]
