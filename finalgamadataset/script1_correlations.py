
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("development_data.csv")

OUTPUT_DIR = "section5b_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = "user_score_mean"

# Make sure the outcome is numeric
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

# ============================================================
# 2. SELECT NUMERICAL PREDICTORS
# ============================================================

# Columns that should NOT be predictors in our primary analysis.
# These may contain direct or indirect information about the outcome.
EXCLUDED_COLUMNS = [
    "game_title",
    "Year",
    "global_user_score_mean",

    # Composite variables excluded pending leakage verification
    "composite_sentiment_correlation_user_mean",
    "composite_sentiment_correlation_global_mean",
    "composite_sentiment_confidence_mean",
    "composite_sentiment_adaptive_mean"
]

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

predictors = [
    col for col in numeric_columns
    if col != TARGET and col not in EXCLUDED_COLUMNS
]

# Convert selected predictors to numeric
X = df[predictors].apply(pd.to_numeric, errors="coerce")

# ============================================================
# 3. CALCULATE PEARSON CORRELATIONS WITH OUTCOME
# ============================================================

correlations = pd.DataFrame({
    "Predictor": X.columns,
    "Correlation": [
        X[col].corr(df[TARGET], method="pearson")
        for col in X.columns
    ],
    "Sample_Size": [
        pd.concat([X[col], df[TARGET]], axis=1).dropna().shape[0]
        for col in X.columns
    ]
})

correlations = correlations.dropna(subset=["Correlation"])

correlations["Absolute_Correlation"] = (
    correlations["Correlation"].abs()
)

correlations = correlations.sort_values(
    "Absolute_Correlation",
    ascending=False
)

print("\n========== ALL CORRELATIONS WITH USER RATINGS ==========")
print(correlations.to_string(index=False))

# ============================================================
# 4. STRONGEST POSITIVE CORRELATIONS
# ============================================================

positive = (
    correlations[correlations["Correlation"] > 0]
    .sort_values("Correlation", ascending=False)
    .head(10)
)

print("\n========== TOP 10 STRONGEST POSITIVE CORRELATIONS ==========")
print(positive.to_string(index=False))

# ============================================================
# 5. STRONGEST NEGATIVE CORRELATIONS
# ============================================================

negative = (
    correlations[correlations["Correlation"] < 0]
    .sort_values("Correlation", ascending=True)
    .head(10)
)

print("\n========== TOP 10 STRONGEST NEGATIVE CORRELATIONS ==========")
print(negative.to_string(index=False))

# ============================================================
# 6. WEAKEST CORRELATIONS
# ============================================================

weakest = (
    correlations
    .sort_values("Absolute_Correlation", ascending=True)
    .head(10)
)

print("\n========== TOP 10 WEAKEST CORRELATIONS ==========")
print(weakest.to_string(index=False))

# ============================================================
# 7. SAVE TABLES
# ============================================================

correlations.to_csv(
    f"{OUTPUT_DIR}/all_correlations.csv",
    index=False
)

positive.to_csv(
    f"{OUTPUT_DIR}/strongest_positive.csv",
    index=False
)

negative.to_csv(
    f"{OUTPUT_DIR}/strongest_negative.csv",
    index=False
)

weakest.to_csv(
    f"{OUTPUT_DIR}/weakest_correlations.csv",
    index=False
)

# ============================================================
# 8. VISUALIZE STRONGEST POSITIVE AND NEGATIVE
# ============================================================

combined = pd.concat([positive, negative])

combined = combined.sort_values("Correlation")

plt.figure(figsize=(11, 8))

plt.barh(
    combined["Predictor"],
    combined["Correlation"]
)

plt.axvline(0, color="black", linewidth=0.8)

plt.xlabel("Pearson Correlation with User Score Mean")
plt.ylabel("Predictor")

plt.title(
    "Strongest Positive and Negative Correlations\n"
    "with User Ratings"
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/strongest_correlations.png",
    dpi=300
)

plt.show()

# ============================================================
# 9. VISUALIZE WEAKEST CORRELATIONS
# ============================================================

weakest_plot = weakest.sort_values("Correlation")

plt.figure(figsize=(11, 7))

plt.barh(
    weakest_plot["Predictor"],
    weakest_plot["Correlation"]
)

plt.axvline(0, color="black", linewidth=0.8)

plt.xlabel("Pearson Correlation with User Score Mean")
plt.ylabel("Predictor")

plt.title("Weakest Linear Correlations with User Ratings")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/weakest_correlations.png",
    dpi=300
)

plt.show()


# ============================================================
# 10. SCATTERPLOTS: METASCORE AND COMPLETION TRIED
# ============================================================

SCATTER_VARIABLES = [
    "metascore",
    "Completion_Tried_It"
]

for col in SCATTER_VARIABLES:

    if col not in df.columns:
        print(f"WARNING: {col} not found. Skipping.")
        continue

    valid = df[[col, TARGET]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()

    if len(valid) < 2 or valid[col].nunique() <= 1:
        print(f"WARNING: Insufficient variation in {col}.")
        continue

    r = valid[col].corr(valid[TARGET])

    plt.figure(figsize=(8, 6))

    plt.scatter(
        valid[col],
        valid[TARGET],
        alpha=0.35
    )

    # Linear trend line
    slope, intercept = np.polyfit(
        valid[col],
        valid[TARGET],
        1
    )

    x_line = np.linspace(
        valid[col].min(),
        valid[col].max(),
        100
    )

    plt.plot(
        x_line,
        slope * x_line + intercept,
        linewidth=2
    )

    plt.xlabel(col)
    plt.ylabel("User Score Mean")

    plt.title(
        f"{col} vs User Ratings\n"
        f"Pearson r = {r:.3f}, n = {len(valid)}"
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/scatter_{col}.png",
        dpi=300
    )

    plt.show()

print("\nAll outcome correlation analyses completed!")
print("Results saved in:", OUTPUT_DIR)
