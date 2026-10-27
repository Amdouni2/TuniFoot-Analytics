from pathlib import Path
import pandas as pd

# Locate the project directory
project_root = Path(__file__).resolve().parents[1]

# Load the feature-engineered dataset
file_path = (
    project_root
    / "data"
    / "processed"
    / "matches_with_features.csv"
)

df = pd.read_csv(file_path)

# Define the input features
features = [
    "home_previous_matches",
    "home_avg_goals_scored",
    "home_avg_goals_conceded",
    "away_previous_matches",
    "away_avg_goals_scored",
    "away_avg_goals_conceded"
]

# Define X: the model's inputs
X = df[features].copy()

# Define y: the outcome we want to predict
y = df["result"].copy()

# Inspect the prepared data
print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nInput features:")
print(X.head())

print("\nTarget values:")
print(y.head())

print("\nMissing values in X:")
print(X.isnull().sum())

print("\nTarget classes:")
print(y.value_counts())

# Convert dates to datetime
df["match_date"] = pd.to_datetime(df["match_date"])

# Sort matches chronologically
df = df.sort_values("match_date").reset_index(drop=True)

# Define X and y after sorting
X = df[features].copy()
y = df["result"].copy()

# Get the unique match dates in chronological order
unique_dates = df["match_date"].drop_duplicates().sort_values().reset_index(drop=True)

# Choose a cutoff date near 80% of the timeline
cutoff_index = int(len(unique_dates) * 0.8)
cutoff_date = unique_dates.iloc[cutoff_index]

# Assign entire dates to either training or testing
train_mask = df["match_date"] < cutoff_date
test_mask = df["match_date"] >= cutoff_date

X_train = X.loc[train_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[train_mask].copy()
y_test = y.loc[test_mask].copy()

# Display split information
print("Cutoff date:", cutoff_date)

print("\nTraining set size:", len(X_train))
print("Testing set size:", len(X_test))

print("\nTraining period:")
print(df.loc[train_mask, "match_date"].min(), "to",
      df.loc[train_mask, "match_date"].max())

print("\nTesting period:")
print(df.loc[test_mask, "match_date"].min(), "to",
      df.loc[test_mask, "match_date"].max())

# Verify that the dates do not overlap
assert (
    df.loc[train_mask, "match_date"].max()
    < df.loc[test_mask, "match_date"].min()
)

print("\nChronological split verified: no overlapping dates.")