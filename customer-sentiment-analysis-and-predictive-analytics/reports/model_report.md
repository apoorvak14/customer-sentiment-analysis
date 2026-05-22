# Model Report

## Objective

The project trains two machine learning models:

- A sentiment classifier that labels feedback as positive, neutral, or negative.
- A churn-risk classifier that predicts whether a customer is likely to churn based on engagement and support metrics.

## Modeling Approach

The sentiment model uses TF-IDF text features and logistic regression. This is a strong baseline for short customer feedback because it is interpretable, fast, and works well with limited data.

The churn model uses standardized numeric customer features and logistic regression. It produces both a binary prediction and a churn probability that can be used for prioritization.

## Recommended Next Steps

- Replace the synthetic sample data with real customer data.
- Add cross-validation and hyperparameter tuning.
- Track model performance by customer segment.
- Add explainability with feature coefficients or SHAP.
- Monitor production drift in sentiment distribution and churn probability.
