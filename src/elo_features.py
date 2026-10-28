from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. Define paths and Elo parameters
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "matches_with_features.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "matches_with_elo.csv"

INITIAL_RATING = 1500
K_FACTOR = 20


# --------------------------------------------------
# 2. Load and prepare the dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

df["match_date"] = pd.to_datetime(df["match_date"])

df = df.sort_values("match_date").reset_index(drop=True)

print("Number of matches:", len(df))


# --------------------------------------------------
# 3. Initialize team ratings
# --------------------------------------------------

ratings = {}

home_elo_values = [None] * len(df)
away_elo_values = [None] * len(df)


# --------------------------------------------------
# 4. Process matches chronologically
# --------------------------------------------------

for match_date, date_matches in df.groupby("match_date", sort=True):

    # Store each team's rating before this date.
    # All matches on the same date use these ratings.
    updates = {}

    for idx, match in date_matches.iterrows():

        home_team = match["home_team"]
        away_team = match["away_team"]

        home_elo = ratings.get(home_team, INITIAL_RATING)
        away_elo = ratings.get(away_team, INITIAL_RATING)

        # Save ratings BEFORE the match.
        home_elo_values[idx] = home_elo
        away_elo_values[idx] = away_elo

        # Expected scores
        expected_home = 1 / (
            1 + 10 ** ((away_elo - home_elo) / 400)
        )

        expected_away = 1 - expected_home

        # Actual scores
        result = match["result"]

        if result == "H":
            actual_home = 1
            actual_away = 0

        elif result == "D":
            actual_home = 0.5
            actual_away = 0.5

        elif result == "A":
            actual_home = 0
            actual_away = 1

        else:
            raise ValueError(
                f"Unexpected result '{result}' at row {idx}"
            )

        # Calculate proposed rating changes.
        # Do not apply them until every match on this date
        # has been processed.
        home_change = K_FACTOR * (
            actual_home - expected_home
        )

        away_change = K_FACTOR * (
            actual_away - expected_away
        )

        updates[home_team] = (
            updates.get(home_team, 0) + home_change
        )

        updates[away_team] = (
            updates.get(away_team, 0) + away_change
        )

    # Apply all updates after processing the entire date.
    for team, change in updates.items():

        current_rating = ratings.get(team, INITIAL_RATING)

        ratings[team] = current_rating + change


# --------------------------------------------------
# 5. Add Elo features to the dataset
# --------------------------------------------------

df["home_elo"] = home_elo_values
df["away_elo"] = away_elo_values

df["elo_difference"] = (
    df["home_elo"] - df["away_elo"]
)


# --------------------------------------------------
# 6. Save the dataset
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\nElo feature generation complete.")

print("Output file:", OUTPUT_FILE)

print("\nNew Elo features:")
print(
    df[
        [
            "match_date",
            "home_team",
            "away_team",
            "result",
            "home_elo",
            "away_elo",
            "elo_difference",
        ]
    ].head(10).to_string(index=False)
)

print("\nDataset shape:", df.shape)