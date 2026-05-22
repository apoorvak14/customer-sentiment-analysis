import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PATH, NUMERIC_FEATURES
from src.data_loader import load_customer_feedback
from src.predict import predict_customer
from src.visualization import plot_churn_by_sentiment, plot_sentiment_distribution

st.set_page_config(page_title="Customer Sentiment Analytics", layout="wide")

st.title("Customer Sentiment Analysis and Predictive Analytics")

data = load_customer_feedback(DATA_PATH)

left, middle, right = st.columns(3)
left.metric("Customers", f"{len(data):,}")
middle.metric("Average NPS", f"{data['nps_score'].mean():.1f}")
right.metric("Churn Rate", f"{data['churned'].mean():.1%}")

tab_overview, tab_predict = st.tabs(["Analytics", "Prediction"])

with tab_overview:
    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.pyplot(plot_sentiment_distribution(data))
    with chart_right:
        st.pyplot(plot_churn_by_sentiment(data))

    st.subheader("Customer Feedback")
    st.dataframe(data, use_container_width=True)

with tab_predict:
    st.subheader("Score a Customer")
    feedback_text = st.text_area("Customer feedback", "The product is easy to use and the support team was excellent.")

    inputs = {}
    col_a, col_b = st.columns(2)
    with col_a:
        inputs["tenure_months"] = st.number_input("Tenure months", min_value=0.0, value=18.0)
        inputs["monthly_spend"] = st.number_input("Monthly spend", min_value=0.0, value=79.0)
        inputs["support_tickets"] = st.number_input("Support tickets", min_value=0.0, value=1.0)
    with col_b:
        inputs["nps_score"] = st.slider("NPS score", 0.0, 10.0, 8.0)
        inputs["usage_score"] = st.slider("Usage score", 0.0, 100.0, 80.0)

    if st.button("Predict"):
        try:
            result = predict_customer(feedback_text, {key: inputs[key] for key in NUMERIC_FEATURES})
            st.success(f"Sentiment: {result['sentiment']}")
            st.metric("Churn Probability", f"{result['churn_probability']:.1%}")
            st.json(result["sentiment_scores"])
        except FileNotFoundError:
            st.warning("Models are not trained yet. Run `python -m src.train` first.")
