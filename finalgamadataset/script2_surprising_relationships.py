
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

# ============================================================
# 2. SELECT PREDICTORS
# ============================================================

EXCLUDED_COLUMNS = [
    "game_title",
    "Year",
    TARGET,
    "global_user_score_mean",

    # Composite variables excluded pending verification
    "composite_sentiment_correlation_user_mean",
    "composite_sentiment_correlation_global_mean",
    "composite_sentiment_confidence_mean",
    "composite_sentiment_adaptive_mean"
]

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

predictors = [
    col for col in numeric_columns
    if col not in EXCLUDED_COLUMNS
]

X = df[predictors].apply(pd.to_numeric, errors="coerce")

# ============================================================
# 3. CALCULATE CORRELATION MATRIX
# ============================================================

corr_matrix = X.corr(method="pearson")

# ============================================================
# 4. EXTRACT UNIQUE VARIABLE PAIRS
# ============================================================

pairs = []

for i in range(len(corr_matrix.columns)):

    for j in range(i + 1, len(corr_matrix.columns)):

        col1 = corr_matrix.columns[i]
        col2 = corr_matrix.columns[j]

        correlation = corr_matrix.iloc[i, j]

        if pd.notna(correlation):

            pairs.append({
                "Variable_1": col1,
                "Variable_2": col2,
                "Correlation": correlation,
                "Absolute_Correlation": abs(correlation)
            })

pairs_df = pd.DataFrame(pairs)

pairs_df = pairs_df.sort_values(
    "Absolute_Correlation",
    ascending=False
)

# ============================================================
# 5. STRONGEST POSITIVE RELATIONSHIPS
# ============================================================

strongest_positive = (
    pairs_df[pairs_df["Correlation"] > 0]
    .sort_values("Correlation", ascending=False)
    .head(15)
)

print("\n========== STRONGEST POSITIVE PREDICTOR PAIRS ==========")
print(strongest_positive.to_string(index=False))

# ============================================================
# 6. STRONGEST NEGATIVE RELATIONSHIPS
# ============================================================

strongest_negative = (
    pairs_df[pairs_df["Correlation"] < 0]
    .sort_values("Correlation")
    .head(15)
)

print("\n========== STRONGEST NEGATIVE PREDICTOR PAIRS ==========")
print(strongest_negative.to_string(index=False))


# ============================================================
# 7. IDENTIFY CROSS-GROUP RELATIONSHIPS
# ============================================================

groups = {
    "Sales": [
        "NA_Sales", "EU_Sales", "JP_Sales",
        "Other_Sales", "Total_Sales"
    ],

    "Reviews": [
        "user_review_count", "critic_review_count"
    ],

    "Difficulty": [
        c for c in predictors
        if c.startswith("Difficulty_")
    ],

    "Playtime": [
        c for c in predictors
        if c.startswith("Play Time_")
    ],

    "Completion": [
        c for c in predictors
        if c.startswith("Completion_")
    ],

    "Critic Score": [
        "metascore", "Avg_Rank"
    ],

    "Sentiment": [
        "vader_sentiment_mean",
        "textblob_sentiment_mean",
        "hf_sentiment_mean"
    ]
}

def get_group(variable):

    for group, columns in groups.items():

        if variable in columns:
            return group

    return "Other"


pairs_df["Group_1"] = (
    pairs_df["Variable_1"].apply(get_group)
)

pairs_df["Group_2"] = (
    pairs_df["Variable_2"].apply(get_group)
)

# Keep relationships between conceptually different groups
cross_group = pairs_df[
    pairs_df["Group_1"] != pairs_df["Group_2"]
].copy()

cross_group = cross_group.sort_values(
    "Absolute_Correlation",
    ascending=False
)

print("\n========== STRONGEST CROSS-GROUP RELATIONSHIPS ==========")

print(
    cross_group.head(30).to_string(index=False)
)

# Specifically inspect sentiment relationships
sentiment_pairs = pairs_df[
    pairs_df["Variable_1"].isin(groups["Sentiment"])
    | pairs_df["Variable_2"].isin(groups["Sentiment"])
].copy()

sentiment_pairs = sentiment_pairs.sort_values(
    "Absolute_Correlation",
    ascending=False
)

print("\n========== ALL SENTIMENT PREDICTOR RELATIONSHIPS ==========")

print(
    sentiment_pairs.to_string(index=False)
)

sentiment_pairs.to_csv(
    f"{OUTPUT_DIR}/sentiment_predictor_correlations.csv",
    index=False
)

# ============================================================
# 8. IDENTIFY RELATIONSHIPS THAT MAY BE UNEXPECTED
# ============================================================

# This is a screening step, NOT an automatic judgment
# that a correlation is surprising.

print("\n========== POTENTIALLY SURPRISING PAIRS TO INVESTIGATE ==========")

for _, row in cross_group.head(15).iterrows():

    print(
        f"{row['Variable_1']} ({row['Group_1']}) "
        f"vs {row['Variable_2']} ({row['Group_2']}): "
        f"r = {row['Correlation']:.3f}"
    )

# ============================================================
# 9. SAVE RESULTS
# ============================================================

pairs_df.to_csv(
    f"{OUTPUT_DIR}/all_predictor_pairs.csv",
    index=False
)

strongest_positive.to_csv(
    f"{OUTPUT_DIR}/positive_predictor_pairs.csv",
    index=False
)

strongest_negative.to_csv(
    f"{OUTPUT_DIR}/negative_predictor_pairs.csv",
    index=False
)

cross_group.to_csv(
    f"{OUTPUT_DIR}/cross_group_correlations.csv",
    index=False
)

# ============================================================
# 10. VISUALIZE CORRELATION MATRIX
# ============================================================

plt.figure(figsize=(17, 14))

im = plt.imshow(
    corr_matrix,
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    aspect="auto"
)

plt.colorbar(im, label="Pearson Correlation")

plt.xticks(
    range(len(corr_matrix.columns)),
    corr_matrix.columns,
    rotation=90,
    fontsize=7
)

plt.yticks(
    range(len(corr_matrix.columns)),
    corr_matrix.columns,
    fontsize=7
)

plt.title("Correlation Matrix of Numerical Predictors")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/full_predictor_heatmap.png",
    dpi=300
)

plt.show()

# ============================================================
# 11. VISUALIZE TOP 15 CROSS-GROUP RELATIONSHIPS
# ============================================================

top_pairs = cross_group.head(15).copy()

top_pairs["Pair"] = (
    top_pairs["Variable_1"]
    + " vs "
    + top_pairs["Variable_2"]
)

top_pairs = top_pairs.sort_values("Correlation")

plt.figure(figsize=(12, 9))

plt.barh(
    top_pairs["Pair"],
    top_pairs["Correlation"]
)

plt.axvline(0, color="black", linewidth=0.8)

plt.xlabel("Pearson Correlation")
plt.title("Strongest Relationships Across Predictor Groups")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/cross_group_correlations.png",
    dpi=300
)

plt.show()

print("\nPredictor relationship analysis completed!")
