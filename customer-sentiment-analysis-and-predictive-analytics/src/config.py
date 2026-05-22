from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "customer_feedback.csv"
MODEL_DIR = PROJECT_ROOT / "models"
SENTIMENT_MODEL_PATH = MODEL_DIR / "sentiment_model.joblib"
CHURN_MODEL_PATH = MODEL_DIR / "churn_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

TEXT_COLUMN = "feedback_text"
SENTIMENT_TARGET = "sentiment"
CHURN_TARGET = "churned"

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_spend",
    "support_tickets",
    "nps_score",
    "usage_score",
]
