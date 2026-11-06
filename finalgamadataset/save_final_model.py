
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor


# --------------------------------------------------
# 1. LOAD DEVELOPMENT DATA
# --------------------------------------------------

df = pd.read_csv("development_data.csv")

TARGET = "user_score_mean"

# Remove rows without a target value, if any.
df = df.dropna(subset=[TARGET])

X = df.drop(columns=[TARGET])
y = df[TARGET]


# --------------------------------------------------
# 2. REMOVE EXCLUDED VARIABLES
# --------------------------------------------------

excluded_columns = [
    "composite_sentiment_correlation_user_mean",
    "composite_sentiment_correlation_global_mean",
    "composite_sentiment_confidence_mean",
    "composite_sentiment_adaptive_mean"
]

identifier_columns = [
    col for col in X.columns
    if col.lower() in [
        "id", "name", "game_id",
        "game_title", "global_user_score_mean"
    ]
    or col.lower().endswith("_id")
    or col.lower().startswith("unnamed:")
]

columns_to_remove = [
    col for col in excluded_columns + identifier_columns
    if col in X.columns
]

X = X.drop(columns=columns_to_remove)

# Feature Set 3 contains all remaining selected predictors.
selected_features = X.columns.tolist()


# --------------------------------------------------
# 3. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# --------------------------------------------------

numerical_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    exclude=["number"]
).columns.tolist()


# --------------------------------------------------
# 4. DEFINE PREPROCESSING
# --------------------------------------------------

# Numerical variables: fill missing values and standardize.
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical variables: fill missing values and one-hot encode.
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])


# --------------------------------------------------
# 5. DEFINE THE SELECTED RANDOM FOREST MODEL
# --------------------------------------------------

random_forest = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=1
)

# Combine preprocessing and the model into one pipeline.
final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", random_forest)
])


# --------------------------------------------------
# 6. FIT ON ALL DEVELOPMENT OBSERVATIONS
# --------------------------------------------------

final_pipeline.fit(X, y)


# --------------------------------------------------
# 7. SAVE THE COMPLETE PIPELINE
# --------------------------------------------------

joblib.dump(
    final_pipeline,
    "final_random_forest_pipeline.joblib"
)

print("Final model successfully trained and saved!")
print("Training observations:", len(X))
print("Number of selected predictors:", len(selected_features))
print("Numerical predictors:", len(numerical_columns))
print("Categorical predictors:", len(categorical_columns))
print("Selected algorithm: Random Forest")
print("Selected feature set: Feature Set 3")
print("Saved file: final_random_forest_pipeline.joblib")
