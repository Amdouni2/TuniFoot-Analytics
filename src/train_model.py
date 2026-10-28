from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import StandardScaler
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
    / "matches_with_elo.csv"
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
    "home_elo",
    "away_elo",
    "elo_difference",
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
# Baseline: Always predict the most frequent class
# --------------------------------------------------

majority_class = y_train.value_counts().idxmax()

baseline_predictions = [
    majority_class
] * len(y_test)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions,
)

baseline_macro_f1 = f1_score(
    y_test,
    baseline_predictions,
    average="macro",
    zero_division=0,
)

print("\nMajority-class baseline")
print("-----------------------")
print("Predicted class:", majority_class)
print(f"Accuracy: {baseline_accuracy:.4f}")
print(f"Macro F1: {baseline_macro_f1:.4f}")


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
# 5. Define Logistic Regression pipeline
# --------------------------------------------------

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),

    ("scaler", StandardScaler()),

    ("classifier", LogisticRegression(
        max_iter=5000,
        random_state=42,
        class_weight="balanced",
    )),
])
# --------------------------------------------------
# 6. Train Logistic Regression
# --------------------------------------------------

pipeline.fit(X_train, y_train)

print("\nLogistic Regression training completed.")


# --------------------------------------------------
# 7. Predict test outcomes
# --------------------------------------------------

y_pred = pipeline.predict(X_test)


# --------------------------------------------------
# 8. Evaluate the model
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
)

print("\nLogistic Regression + Elo")
print("-------------------------")

print(f"Test accuracy: {accuracy:.4f}")
print(f"Test accuracy percentage: {accuracy * 100:.2f}%")
print(f"Macro F1-score: {macro_f1:.4f}")

print("\nF1-score (Home win):",
      f1_score(y_test, y_pred, labels=["H"], average="macro"))

print("F1-score (Draw):",
      f1_score(y_test, y_pred, labels=["D"], average="macro"))

print("F1-score (Away win):",
      f1_score(y_test, y_pred, labels=["A"], average="macro"))

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
# 9. Display confusion matrix
# --------------------------------------------------

ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    labels=["H", "D", "A"],
    display_labels=[
        "Home win",
        "Draw",
        "Away win",
    ],
    cmap="Blues",
    values_format="d",
)

plt.title("TuniFoot Analytics - Logistic Regression + Elo")

plt.tight_layout()

figures_dir = project_root / "reports" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(
    figures_dir / "logistic_regression_elo_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()



#F1-score without recent form features = 0.4395 and test accuracy = 0.4666

#F1-score with recent form features = 0.4534 and test accuracy = 0.4816

#F1-score with only recent form features = 0.4080 and test accuracy = 0.4274

#F1-score with elo = 0.4595 and test accuracy = 0.4804

#F1-score LogisticRegression with elo = 0.4460 and test accuracy = 0.4965
