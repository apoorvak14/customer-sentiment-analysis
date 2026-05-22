import json

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CHURN_MODEL_PATH, METRICS_PATH, MODEL_DIR, NUMERIC_FEATURES, SENTIMENT_MODEL_PATH
from src.data_loader import load_customer_feedback, split_churn_data, split_sentiment_data


def build_sentiment_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def build_churn_model() -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[("numeric", StandardScaler(), NUMERIC_FEATURES)],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessing),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def train_models():
    data = load_customer_feedback()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    text_x, sentiment_y = split_sentiment_data(data)
    churn_x, churn_y = split_churn_data(data)

    text_train, text_test, sentiment_train, sentiment_test = train_test_split(
        text_x,
        sentiment_y,
        test_size=0.25,
        random_state=42,
        stratify=sentiment_y,
    )

    churn_train, churn_test, churn_train_y, churn_test_y = train_test_split(
        churn_x,
        churn_y,
        test_size=0.25,
        random_state=42,
        stratify=churn_y,
    )

    sentiment_model = build_sentiment_model()
    sentiment_model.fit(text_train, sentiment_train)
    sentiment_predictions = sentiment_model.predict(text_test)

    churn_model = build_churn_model()
    churn_model.fit(churn_train, churn_train_y)
    churn_predictions = churn_model.predict(churn_test)
    churn_probabilities = churn_model.predict_proba(churn_test)[:, 1]

    metrics = {
        "sentiment": {
            "accuracy": accuracy_score(sentiment_test, sentiment_predictions),
            "macro_f1": f1_score(sentiment_test, sentiment_predictions, average="macro"),
            "classification_report": classification_report(
                sentiment_test,
                sentiment_predictions,
                output_dict=True,
                zero_division=0,
            ),
        },
        "churn": {
            "accuracy": accuracy_score(churn_test_y, churn_predictions),
            "roc_auc": roc_auc_score(churn_test_y, churn_probabilities),
            "classification_report": classification_report(
                churn_test_y,
                churn_predictions,
                output_dict=True,
                zero_division=0,
            ),
        },
    }

    joblib.dump(sentiment_model, SENTIMENT_MODEL_PATH)
    joblib.dump(churn_model, CHURN_MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def main():
    metrics = train_models()
    print("Training complete.")
    print(f"Sentiment accuracy: {metrics['sentiment']['accuracy']:.3f}")
    print(f"Churn ROC AUC: {metrics['churn']['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
