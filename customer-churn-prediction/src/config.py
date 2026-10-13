from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "reports" / "figures"
MLFLOW_URI = f"sqlite:///{ARTIFACTS / 'mlflow.db'}"
MLFLOW_ARTIFACTS = str(ARTIFACTS / "mlruns")
SEED = 42
for p in (ARTIFACTS, FIGURES):
    p.mkdir(parents=True, exist_ok=True)
