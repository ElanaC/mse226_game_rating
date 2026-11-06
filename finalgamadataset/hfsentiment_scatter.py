import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("development_data.csv")

x_col = "hf_sentiment_mean"
y_col = "user_score_mean"

# Remove missing observations
plot_data = df[[x_col, y_col]].dropna()

# Calculate Pearson correlation
r = plot_data[x_col].corr(plot_data[y_col])

# Scatterplot
plt.figure(figsize=(9, 6))

plt.scatter(
    plot_data[x_col],
    plot_data[y_col],
    alpha=0.4
)

# Linear trend line
slope, intercept = np.polyfit(
    plot_data[x_col],
    plot_data[y_col],
    1
)

x_line = np.linspace(
    plot_data[x_col].min(),
    plot_data[x_col].max(),
    100
)

plt.plot(
    x_line,
    slope * x_line + intercept,
    linewidth=2
)

plt.xlabel("HF Sentiment Mean")
plt.ylabel("User Score Mean")

plt.title(
    f"HF Sentiment Mean vs. User Ratings\n"
    f"Pearson r = {r:.3f}, n = {len(plot_data):,}"
)

plt.tight_layout()

plt.savefig(
    "section5b_results/scatter_hf_sentiment.png",
    dpi=300
)

plt.show()