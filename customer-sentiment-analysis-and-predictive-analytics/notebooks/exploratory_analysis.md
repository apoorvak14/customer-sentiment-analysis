# Exploratory Analysis Notes

Use this notebook-style guide to explore the dataset before training.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("../data/customer_feedback.csv")
data.head()
data.describe()
data["sentiment"].value_counts()
data.groupby("sentiment")["churned"].mean()

sns.scatterplot(data=data, x="usage_score", y="nps_score", hue="churned")
plt.show()
```

Questions to investigate:

- Are negative comments associated with higher churn?
- Do high support ticket counts signal churn risk?
- Which customer segments have lower NPS or usage?
- Are there feedback themes that appear before churn?
