from pathlib import Path

import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
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
# 4. Create the machine-learning pipeline
# --------------------------------------------------

model = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)


# --------------------------------------------------
# 5. Train the model
# --------------------------------------------------

model.fit(X_train, y_train)

print("\nModel training completed.")


# --------------------------------------------------
# 6. Predict test outcomes
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# 7. Evaluate the model
# --------------------------------------------------

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

plt.title("TuniFoot Analytics - Logistic Regression")
plt.tight_layout()

figures_dir = project_root / "reports" / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(
    figures_dir / "logistic_regression_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()