
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

df = pd.read_csv("development_data.csv")

TARGET = "user_score_mean"

# Remove rows without a target value, if any.
df = df.dropna(subset=[TARGET])

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Exclude identifiers and composite sentiment variables.
excluded_columns = [
    "composite_sentiment_correlation_user_mean",
    "composite_sentiment_correlation_global_mean",
    "composite_sentiment_confidence_mean",
    "composite_sentiment_adaptive_mean"
]

# Also exclude columns that function as identifiers.
identifier_columns = [
    col for col in X.columns
    if col.lower() in ["id", "name", "game_id","game_title", "global_user_score_mean"]
    or col.lower().endswith("_id")
    or col.lower().startswith("unnamed:")
]

X = X.drop(
    columns=[
        col for col in excluded_columns + identifier_columns
        if col in X.columns
    ]
)


# --------------------------------------------------
# 2. DEFINE FEATURE SETS
# --------------------------------------------------

# Feature Set 1: Basic game characteristics.
game_characteristics = [
    "Genre",
    "platforms",
    "developer",
    "Publisher"
]

# Keep only columns that actually exist in the dataset.
feature_set_1 = [
    col for col in game_characteristics
    if col in X.columns
]

# Feature Set 2: Add commercial and review information.
commercial_review_candidates = [
    "metascore",
    "user_review_count",
    "critic_review_count",
    "NA_Sales",
    "EU_Sales",
    "Other_Sales",
    "Total_Sales",
    "Avg_Rank"
]

feature_set_2_additions = [
    col for col in commercial_review_candidates
    if col in X.columns
]

feature_set_2 = list(dict.fromkeys(
    feature_set_1 + feature_set_2_additions
))

# Feature Set 3: All remaining selected predictors.
# This includes sentiment, completion, difficulty,
# play-time, and other available non-identifier variables.
feature_set_3 = list(X.columns)

feature_sets = {
    "Feature Set 1 - Game Characteristics": feature_set_1,
    "Feature Set 2 - Commercial and Review": feature_set_2,
    "Feature Set 3 - All Selected Predictors": feature_set_3
}

print("\n========== FEATURE SETS ==========")

for name, columns in feature_sets.items():
    print(f"\n{name}")
    print(f"Number of predictors: {len(columns)}")
    print(columns)


# --------------------------------------------------
# 3. TRAIN / VALIDATION SPLIT
# --------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n========== DATA SPLIT ==========")
print("Training observations:", len(X_train))
print("Validation observations:", len(X_val))


# --------------------------------------------------
# 4. DEFINE ALGORITHMS
# --------------------------------------------------

models = {
    "Linear Regression": LinearRegression(),

    "Ridge Regression": Ridge(alpha=1.0),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# --------------------------------------------------
# 5. PREPROCESSING FUNCTION
# --------------------------------------------------

def create_pipeline(model, selected_columns):

    # Select predictors for this feature set.
    X_train_selected = X_train[selected_columns]

    # Identify numerical and categorical columns.
    numerical_columns = X_train_selected.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X_train_selected.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    # Numerical preprocessing:
    # Fill missing values with the median, then standardize.
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing:
    # Fill missing values with the most common category,
    # then convert categories into numerical indicator columns.
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    # Apply the appropriate preprocessing to each column type.
    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    # Combine preprocessing and prediction into one pipeline.
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline


# --------------------------------------------------
# 6. FIVE-FOLD CROSS-VALIDATION + VALIDATION RESULTS
# --------------------------------------------------

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

print("\n========== MODEL COMPARISON ==========")

for feature_name, selected_columns in feature_sets.items():

    # Skip a feature set if no predictors were found.
    if not selected_columns:
        print(f"\nSkipping {feature_name}: no matching columns.")
        continue

    for model_name, model in models.items():

        pipeline = create_pipeline(
            model,
            selected_columns
        )

        # Five-fold cross-validation on the training data.
        cv_scores = cross_val_score(
            pipeline,
            X_train[selected_columns],
            y_train,
            cv=kf,
            scoring="neg_mean_absolute_error",
            n_jobs=1
        )

        cv_mae = -cv_scores.mean()

        # Fit the pipeline on the entire training portion.
        pipeline.fit(
            X_train[selected_columns],
            y_train
        )

        # Predict ratings for the validation observations.
        predictions = pipeline.predict(
            X_val[selected_columns]
        )

        # Calculate validation metrics.
        val_mae = mean_absolute_error(
            y_val,
            predictions
        )

        val_rmse = np.sqrt(
            mean_squared_error(y_val, predictions)
        )

        val_r2 = r2_score(
            y_val,
            predictions
        )

        results.append({
            "Feature Set": feature_name,
            "Algorithm": model_name,
            "CV MAE": cv_mae,
            "Validation MAE": val_mae,
            "Validation RMSE": val_rmse,
            "Validation R2": val_r2
        })

        print(
            f"\n{feature_name} | {model_name}"
        )
        print(f"CV MAE: {cv_mae:.4f}")
        print(f"Validation MAE: {val_mae:.4f}")
        print(f"Validation RMSE: {val_rmse:.4f}")
        print(f"Validation R2: {val_r2:.4f}")


# --------------------------------------------------
# 7. DISPLAY AND SAVE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Validation MAE",
    ascending=True
)

print("\n========== FINAL RESULTS (SORTED BY MAE) ==========")
print(results_df.to_string(index=False))

results_df.to_csv(
    "model_comparison_results.csv",
    index=False
)

print("\nResults saved to model_comparison_results.csv")
