import pandas as pd

from src.config import CHURN_TARGET, DATA_PATH, NUMERIC_FEATURES, SENTIMENT_TARGET, TEXT_COLUMN


def load_customer_feedback(path=DATA_PATH) -> pd.DataFrame:
    """Load and validate the customer feedback dataset."""
    data = pd.read_csv(path)
    required_columns = {TEXT_COLUMN, SENTIMENT_TARGET, CHURN_TARGET, *NUMERIC_FEATURES}
    missing = required_columns.difference(data.columns)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Dataset is missing required columns: {missing_list}")
    return data


def split_sentiment_data(data: pd.DataFrame):
    return data[TEXT_COLUMN].fillna(""), data[SENTIMENT_TARGET]


def split_churn_data(data: pd.DataFrame):
    return data[NUMERIC_FEATURES], data[CHURN_TARGET]
