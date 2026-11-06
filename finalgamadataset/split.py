
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the original dataset
df = pd.read_csv("completed_records_plus_sentiment_metrics.csv")

# Fixed random seed for reproducibility
SEED = 42

# Randomly split into 80% development and 20% holdout
development_data, holdout_data = train_test_split(
    df,
    test_size=0.20,
    random_state=SEED
)

# Save both datasets
development_data.to_csv("development_data.csv", index=False)
holdout_data.to_csv("holdout_data.csv", index=False)

# Verify sample sizes
print("Original dataset:", len(df))
print("Development data:", len(development_data))
print("Holdout data:", len(holdout_data))
print("Random seed:", SEED)
