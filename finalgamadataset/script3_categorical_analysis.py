
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.stats import kruskal

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("development_data.csv")

OUTPUT_DIR = "section5b_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET = "user_score_mean"

df[TARGET] = pd.to_numeric(
    df[TARGET],
    errors="coerce"
)

# ============================================================
# 2. DEFINE CATEGORICAL VARIABLES
# ============================================================

categorical_columns = [
    "Genre",
    "platforms",
    "developer",
    "Publisher"
]

# Minimum number of games required for a category
MIN_GROUP_SIZE = 20

# Limit the number of groups displayed for high-cardinality
# variables such as developer and Publisher.
TOP_N = 15

# ============================================================
# 3. CLEAN CATEGORICAL DATA
# ============================================================

for col in categorical_columns:

    df[col] = (
        df[col]
        .astype("string")
        .str.strip()
    )

    df[col] = df[col].replace(
        ["", "nan", "None", "NA"],
        pd.NA
    )

# ============================================================
# 4. CREATE SUMMARY TABLES
# ============================================================

all_summaries = []

for col in categorical_columns:

    print("\n" + "=" * 70)
    print(f"ANALYSIS OF: {col}")
    print("=" * 70)

    # Remove rows without outcome or category
    subset = df[[col, TARGET]].dropna()

    # Calculate statistics for every category
    summary = (
        subset.groupby(col)[TARGET]
        .agg(
            Count="count",
            Mean="mean",
            Median="median",
            Std="std",
            Min="min",
            Max="max"
        )
        .reset_index()
    )

    # Retain categories with enough observations
    summary = summary[
        summary["Count"] >= MIN_GROUP_SIZE
    ]

    # Sort by average user rating
    summary = summary.sort_values(
        "Mean",
        ascending=False
    )

    # Limit displayed categories to top N by sample size
    if col != "Genre":

        selected_categories = (
            summary.sort_values(
                "Count",
                ascending=False
            )
            .head(TOP_N)[col]
            .tolist()
        )

        summary = summary[
            summary[col].isin(selected_categories)
        ]

        summary = summary.sort_values(
            "Mean",
            ascending=False
        )

    print("\nRating statistics by category:")
    print(summary.to_string(index=False))

    # Save individual summary
    summary.to_csv(
        f"{OUTPUT_DIR}/{col}_rating_summary.csv",
        index=False
    )

    all_summaries.append(
        summary.assign(Category_Type=col)
    )

    # ========================================================
    # 5. BOXPLOT: USER RATING DISTRIBUTIONS
    # ========================================================

    categories = summary[col].tolist()

    box_data = [
        subset.loc[
            subset[col] == category,
            TARGET
        ].values

        for category in categories
    ]

    if len(box_data) >= 2:

        plt.figure(figsize=(13, 7))

        plt.boxplot(
            box_data,
            tick_labels=categories,
            showfliers=False
        )

        plt.xticks(
            rotation=40,
            ha="right"
        )

        plt.ylabel("User Score Mean")
        plt.xlabel(col)

        plt.title(
            f"Distribution of User Ratings by {col}"
        )

        plt.tight_layout()

        safe_name = col.replace(" ", "_")

        plt.savefig(
            f"{OUTPUT_DIR}/{safe_name}_boxplot.png",
            dpi=300
        )

        plt.show()

    # ========================================================
    # 6. BAR CHART: AVERAGE USER RATINGS
    # ========================================================

    plot_summary = summary.sort_values("Mean")

    plt.figure(figsize=(11, 7))

    plt.barh(
        plot_summary[col].astype(str),
        plot_summary["Mean"]
    )

    plt.xlabel("Average User Score Mean")
    plt.ylabel(col)

    plt.title(
        f"Average User Ratings by {col}"
    )

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/{safe_name}_mean_ratings.png",
        dpi=300
    )

    plt.show()

    # ========================================================
    # 7. KRUSKAL-WALLIS TEST
    # ========================================================

    groups = []

    for category in categories:

        values = subset.loc[
            subset[col] == category,
            TARGET
        ].dropna().values

        if len(values) >= MIN_GROUP_SIZE:
            groups.append(values)

    if len(groups) >= 2:

        statistic, p_value = kruskal(*groups)

        print("\nKruskal-Wallis test:")
        print("H statistic:", round(statistic, 4))
        print("P-value:", p_value)

        if p_value < 0.05:

            print(
                "Result: Evidence that at least one "
                "category differs in its rating distribution."
            )

        else:

            print(
                "Result: Insufficient evidence of differences "
                "across the compared categories."
            )

# ============================================================
# 8. COMBINE ALL CATEGORY SUMMARIES
# ============================================================

combined_summary = pd.concat(
    all_summaries,
    ignore_index=True
)

combined_summary.to_csv(
    f"{OUTPUT_DIR}/all_categorical_summaries.csv",
    index=False
)

print("\n" + "=" * 70)
print("CATEGORICAL ANALYSIS COMPLETED!")
print("=" * 70)

print("All summary tables and plots saved in:", OUTPUT_DIR)
