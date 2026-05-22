# Customer Sentiment Analysis and Predictive Analytics

A complete machine learning project for analyzing customer feedback sentiment and predicting customer outcomes from support, product, and engagement signals.

## Features

- Text sentiment classification for customer reviews and support comments
- Predictive churn-risk modeling from customer behavior metrics
- Synthetic sample dataset for immediate experimentation
- Training pipeline with saved model artifacts
- CLI prediction script
- Streamlit dashboard for interactive analysis
- Basic tests for data loading and model workflows

## Project Structure

```text
customer-sentiment-analysis-and-predictive-analytics/
├── app/
│   └── streamlit_app.py
├── data/
│   └── customer_feedback.csv
├── models/
│   └── .gitkeep
├── notebooks/
│   └── exploratory_analysis.md
├── reports/
│   └── model_report.md
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── predict.py
│   ├── train.py
│   └── visualization.py
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## Train Models

```bash
python -m src.train
```

This creates model artifacts in `models/`:

- `sentiment_model.joblib`
- `churn_model.joblib`
- `metrics.json`

## Run Predictions

```bash
python -m src.predict --text "The support team solved my issue quickly" --tenure 18 --monthly-spend 79 --tickets 1 --nps 9 --usage 84
```

## Launch Dashboard

```bash
streamlit run app/streamlit_app.py
```

## Run Tests

```bash
pytest
```

## Dataset Columns

- `customer_id`: Unique customer identifier
- `feedback_text`: Review or support comment
- `sentiment`: Target sentiment label: positive, neutral, or negative
- `tenure_months`: Customer tenure
- `monthly_spend`: Average monthly spend
- `support_tickets`: Number of support tickets in the recent period
- `nps_score`: Net Promoter Score from 0 to 10
- `usage_score`: Product usage score from 0 to 100
- `churned`: Target churn indicator, 1 if churned and 0 otherwise

## Notes

The included dataset is synthetic and intended for demonstration. Replace `data/customer_feedback.csv` with real customer data before production use.
