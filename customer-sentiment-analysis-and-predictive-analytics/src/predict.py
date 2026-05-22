import argparse

import joblib
import pandas as pd

from src.config import CHURN_MODEL_PATH, NUMERIC_FEATURES, SENTIMENT_MODEL_PATH


def load_models():
    if not SENTIMENT_MODEL_PATH.exists() or not CHURN_MODEL_PATH.exists():
        raise FileNotFoundError("Model artifacts not found. Run `python -m src.train` first.")
    return joblib.load(SENTIMENT_MODEL_PATH), joblib.load(CHURN_MODEL_PATH)


def predict_customer(text: str, numeric_features: dict):
    sentiment_model, churn_model = load_models()
    sentiment = sentiment_model.predict([text])[0]
    sentiment_scores = dict(zip(sentiment_model.classes_, sentiment_model.predict_proba([text])[0]))

    feature_frame = pd.DataFrame([{name: numeric_features[name] for name in NUMERIC_FEATURES}])
    churn_probability = churn_model.predict_proba(feature_frame)[0][1]
    churn_prediction = int(churn_probability >= 0.5)

    return {
        "sentiment": sentiment,
        "sentiment_scores": sentiment_scores,
        "churn_probability": churn_probability,
        "churn_prediction": churn_prediction,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Predict customer sentiment and churn risk.")
    parser.add_argument("--text", required=True, help="Customer feedback text.")
    parser.add_argument("--tenure", type=float, required=True, help="Customer tenure in months.")
    parser.add_argument("--monthly-spend", type=float, required=True, help="Average monthly spend.")
    parser.add_argument("--tickets", type=float, required=True, help="Recent support ticket count.")
    parser.add_argument("--nps", type=float, required=True, help="NPS score from 0 to 10.")
    parser.add_argument("--usage", type=float, required=True, help="Usage score from 0 to 100.")
    return parser.parse_args()


def main():
    args = parse_args()
    result = predict_customer(
        args.text,
        {
            "tenure_months": args.tenure,
            "monthly_spend": args.monthly_spend,
            "support_tickets": args.tickets,
            "nps_score": args.nps,
            "usage_score": args.usage,
        },
    )
    print(f"Sentiment: {result['sentiment']}")
    print(f"Churn probability: {result['churn_probability']:.2%}")
    print(f"Churn prediction: {result['churn_prediction']}")


if __name__ == "__main__":
    main()
