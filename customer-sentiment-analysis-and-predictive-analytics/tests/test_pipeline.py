from src.data_loader import load_customer_feedback, split_churn_data, split_sentiment_data
from src.train import build_churn_model, build_sentiment_model


def test_load_customer_feedback_has_required_columns():
    data = load_customer_feedback()
    assert "feedback_text" in data.columns
    assert "sentiment" in data.columns
    assert "churned" in data.columns
    assert len(data) > 0


def test_split_helpers_return_matching_lengths():
    data = load_customer_feedback()
    text_x, sentiment_y = split_sentiment_data(data)
    churn_x, churn_y = split_churn_data(data)
    assert len(text_x) == len(sentiment_y)
    assert len(churn_x) == len(churn_y)


def test_models_can_fit_small_sample():
    data = load_customer_feedback()
    sentiment_model = build_sentiment_model()
    churn_model = build_churn_model()

    sentiment_model.fit(data["feedback_text"], data["sentiment"])
    churn_model.fit(data[["tenure_months", "monthly_spend", "support_tickets", "nps_score", "usage_score"]], data["churned"])

    assert len(sentiment_model.predict(["The service was great"])) == 1
    assert len(churn_model.predict(data[["tenure_months", "monthly_spend", "support_tickets", "nps_score", "usage_score"]].head(1))) == 1
