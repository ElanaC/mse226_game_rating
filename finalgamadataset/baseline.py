
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 1. LOAD DEVELOPMENT DATA
# ============================================================

df = pd.read_csv("development_data.csv")

TARGET = "user_score_mean"

# Remove rows with missing target values, if any
df = df.dropna(subset=[TARGET])

X = df.drop(columns=[TARGET])
y = df[TARGET]

# ============================================================
# 2. TRAIN / VALIDATION SPLIT (80/20)
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training observations:", len(y_train))
print("Validation observations:", len(y_val))

# ============================================================
# 3. FIVE-FOLD CROSS-VALIDATION ON TRAINING DATA
# ============================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_maes = []

print("\n========== 5-FOLD CROSS-VALIDATION ==========")

for fold, (train_idx, val_idx) in enumerate(
    kf.split(X_train), start=1
):

    # Divide training data into fold training/validation
    y_fold_train = y_train.iloc[train_idx]
    y_fold_val = y_train.iloc[val_idx]

    # Predict the median of the fold's training data
    fold_prediction = np.median(y_fold_train)

    predictions = np.full(
        len(y_fold_val),
        fold_prediction
    )

    fold_mae = mean_absolute_error(
        y_fold_val,
        predictions
    )

    cv_maes.append(fold_mae)

    print(f"Fold {fold} MAE: {fold_mae:.4f}")

mean_cv_mae = np.mean(cv_maes)

print("\nAverage 5-Fold CV MAE:", round(mean_cv_mae, 4))

# ============================================================
# 4. FIT BASELINE ON ENTIRE TRAINING SET
# ============================================================

baseline_prediction = np.median(y_train)

# Predict the same median for all validation observations
y_pred_baseline = np.full(
    len(y_val),
    baseline_prediction
)

# ============================================================
# 5. EVALUATE ON VALIDATION SET
# ============================================================

mae = mean_absolute_error(y_val, y_pred_baseline)

rmse = np.sqrt(
    mean_squared_error(y_val, y_pred_baseline)
)

r2 = r2_score(y_val, y_pred_baseline)

print("\n========== FINAL BASELINE VALIDATION RESULTS ==========")

print("Median prediction:", round(baseline_prediction, 4))
print("Validation MAE:", round(mae, 4))
print("Validation RMSE:", round(rmse, 4))
print("Validation R²:", round(r2, 4))

print("\nBaseline evaluation completed!")
