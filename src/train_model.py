from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.impute import SimpleImputer
#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
#from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Load the dataset
# --------------------------------------------------

project_root = Path(__file__).resolve().parents[1]

file_path = (
    project_root
    / "data"
    / "processed"
    / "matches_with_features.csv"
)

df = pd.read_csv(file_path)

df["match_date"] = pd.to_datetime(df["match_date"])

df = df.sort_values("match_date").reset_index(drop=True)


# --------------------------------------------------
# 2. Define features and target
# --------------------------------------------------

features = [
    "home_previous_matches",
    "home_avg_goals_scored",
    "home_avg_goals_conceded",
    "away_previous_matches",
    "away_avg_goals_scored",
    "away_avg_goals_conceded",
    "home_points_per_match",
    "away_points_per_match",
    "home_avg_goal_difference",
    "away_avg_goal_difference",
    "home_last5_points",
    "away_last5_points",
    "home_last5_goal_difference",
    "away_last5_goal_difference",
]

X = df[features].copy()
y = df["result"].copy()


# --------------------------------------------------
# 3. Split chronologically by date
# --------------------------------------------------

unique_dates = (
    df["match_date"]
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)

cutoff_index = int(len(unique_dates) * 0.8)
cutoff_date = unique_dates.iloc[cutoff_index]

train_mask = df["match_date"] < cutoff_date
test_mask = df["match_date"] >= cutoff_date

X_train = X.loc[train_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[train_mask].copy()
y_test = y.loc[test_mask].copy()

print("Cutoff date:", cutoff_date)
print("Training matches:", len(X_train))
print("Testing matches:", len(X_test))

print(
    "Training period:",
    df.loc[train_mask, "match_date"].min(),
    "to",
    df.loc[train_mask, "match_date"].max(),
)

print(
    "Testing period:",
    df.loc[test_mask, "match_date"].min(),
    "to",
    df.loc[test_mask, "match_date"].max(),
)


# --------------------------------------------------
# 4. Prepare chronological cross-validation
# --------------------------------------------------

# Reset indices so that cross-validation indices match X_train and y_train.
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

# Keep the training dates aligned with the training rows.
train_dates = (
    df.loc[train_mask, "match_date"]
    .reset_index(drop=True)
)

# Extract unique dates in chronological order.
unique_train_dates = (
    train_dates
    .drop_duplicates()
    .sort_values()
    .reset_index(drop=True)
)

# Split unique dates, not individual matches.
time_splitter = TimeSeriesSplit(n_splits=4)

# Convert date-based splits into row indices for RandomizedSearchCV.
cv_splits = []

for fold_train_idx, fold_val_idx in time_splitter.split(unique_train_dates):

    fold_train_dates = unique_train_dates.iloc[fold_train_idx]
    fold_val_dates = unique_train_dates.iloc[fold_val_idx]

    row_train_idx = np.flatnonzero(
        train_dates.isin(fold_train_dates).to_numpy()
    )

    row_val_idx = np.flatnonzero(
        train_dates.isin(fold_val_dates).to_numpy()
    )

    cv_splits.append((row_train_idx, row_val_idx))

print("\nChronological cross-validation:")
for fold, (fold_train_idx, fold_val_idx) in enumerate(cv_splits, start=1):
    print(
        f"Fold {fold}: "
        f"{len(fold_train_idx)} training matches, "
        f"{len(fold_val_idx)} validation matches"
    )


# --------------------------------------------------
# 5. Define the model and hyperparameter search
# --------------------------------------------------

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        random_state=42,
        n_jobs=1,
    )),
])

# Hyperparameter combinations to explore.
param_distributions = {
    "classifier__n_estimators": [200, 300, 500],
    "classifier__max_depth": [10, 15, 20, None],
    "classifier__min_samples_split": [5, 10, 20],
    "classifier__min_samples_leaf": [2, 4, 8],
    "classifier__class_weight": [
        "balanced",
        "balanced_subsample",
    ],
}

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=20,
    scoring="f1_macro",
    cv=cv_splits,
    random_state=42,
    n_jobs=-1,
    verbose=2,
    refit=True,
)

# --------------------------------------------------
# 6. Tune and train the best model
# --------------------------------------------------

search.fit(X_train, y_train)

# RandomizedSearchCV automatically refits the best configuration
# on the entire training dataset.
model = search.best_estimator_

print("\nHyperparameter tuning completed.")

print("\nBest hyperparameters:")
for parameter, value in search.best_params_.items():
    print(f"{parameter}: {value}")

print(f"\nBest cross-validation macro F1-score: {search.best_score_:.4f}")
# --------------------------------------------------
# 7. Predict test outcomes
# --------------------------------------------------

y_pred = model.predict(X_test)
# --------------------------------------------------
# 8. Evaluate the model
# --------------------------------------------------
# Calculate the F1-score for each class
print("F1-score (Home win):", f1_score(y_test, y_pred, labels=["H"], average="macro"))
print("F1-score (Draw):", f1_score(y_test, y_pred, labels=["D"], average="macro"))
print("F1-score (Away win):", f1_score(y_test, y_pred, labels=["A"], average="macro"))

# Calculate the overall macro F1-score
macro_f1 = f1_score(y_test, y_pred, average="macro")
print(f"\nMacro F1-score: {macro_f1:.4f}")
accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest accuracy: {accuracy:.4f}")
print(f"Test accuracy percentage: {accuracy * 100:.2f}%")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=["H", "D", "A"],
        target_names=[
            "Home win",
            "Draw",
            "Away win",
        ],
        zero_division=0,
    )
)


# --------------------------------------------------
# 8. Display the confusion matrix
# --------------------------------------------------

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    labels=["H", "D", "A"],
    display_labels=["Home win", "Draw", "Away win"],
    cmap="Blues",
    values_format="d",
)

plt.title("TuniFoot Analytics - Tuned Random Forest")
plt.tight_layout()

figures_dir = project_root / "reports" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(
    figures_dir / "random_forest_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()