import matplotlib.pyplot as plt
import seaborn as sns


def plot_sentiment_distribution(data):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=data, x="sentiment", order=["negative", "neutral", "positive"], ax=ax)
    ax.set_title("Customer Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Customers")
    return fig


def plot_churn_by_sentiment(data):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=data, x="sentiment", y="churned", order=["negative", "neutral", "positive"], ax=ax)
    ax.set_title("Average Churn Rate by Sentiment")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Churn Rate")
    return fig
