"""Elo ratings with home advantage, season regression and goal-difference scaling.

Improvements over a plain Elo:
  - Home advantage (HFA): home team gets a rating bonus when computing expectations.
  - Season regression: ratings are pulled toward the mean between seasons.
  - Promoted teams (new to the data after the first season) start below average.
  - Goal-difference multiplier: big wins move ratings more than narrow ones.
  - K, HFA, regression and promoted start are tuned on OLD seasons only, so the
    seasons used later for walk-forward testing never influence the settings.

Ratings stored for each match are the ratings BEFORE that match. Matches on the
same date all use ratings from before that date.

Run from the project root (after feature_engineering.py):
    python src/elo_features.py
"""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------
# Settings
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "matches_with_features_claude.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "matches_with_elo_claude.csv"

INITIAL_RATING = 1500

# Defaults, used when TUNE is False.
DEFAULT_PARAMS = {
    "k": 20,
    "hfa": 60,
    "regress": 0.25,
    "promo_start": 1400,
    "goal_mult": True,
}

TUNE = True
# Must match N_TEST_SEASONS in train_model.py: these latest seasons are never used for tuning.
N_HOLDOUT_SEASONS = 10
WARMUP_SEASONS = 2  # early seasons are used to build ratings but not scored for tuning

GRID = {
    "k": [10, 20, 30, 40],
    "hfa": [0, 50, 100],
    "regress": [0.0, 0.25, 0.5],
    "promo_start": [1500, 1400],
}


# --------------------------------------------------
# Data preparation
# --------------------------------------------------

def load_matches():
    df = pd.read_csv(INPUT_FILE)
    required = {"match_date", "home_team", "away_team", "home_score", "away_score", "season"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values("match_date", kind="mergesort").reset_index(drop=True)

    # Order seasons by their first match date (not alphabetically).
    season_order = df.groupby("season")["match_date"].min().sort_values().index.tolist()
    season_index = {season: i for i, season in enumerate(season_order)}
    df["_season_idx"] = df["season"].map(season_index)

    print("Number of matches:", len(df))
    print(f"Seasons: {len(season_order)} ({season_order[0]} to {season_order[-1]})")
    return df, season_order


def build_days(df):
    """Group matches by date: list of (season_idx, [(row, home, away, hs, as), ...])."""
    days = []
    for _, day in df.groupby("match_date", sort=True):
        matches = list(zip(
            day.index,
            day["home_team"],
            day["away_team"],
            day["home_score"],
            day["away_score"],
        ))
        days.append((int(day["_season_idx"].iloc[0]), matches))
    return days


# --------------------------------------------------
# Elo engine
# --------------------------------------------------

def goal_multiplier(goal_diff):
    """Larger margins move ratings more (common football-Elo scaling)."""
    goal_diff = abs(goal_diff)
    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    return (11 + goal_diff) / 8


def run_elo(days, n_rows, k, hfa, regress, promo_start, goal_mult):
    ratings = {}
    home_elo = np.full(n_rows, np.nan)
    away_elo = np.full(n_rows, np.nan)
    expected_home = np.full(n_rows, np.nan)
    actual_home = np.full(n_rows, np.nan)

    current_season = None

    for season_idx, matches in days:

        # New season: pull every rating toward the mean.
        if current_season is not None and season_idx != current_season and regress > 0:
            for team in ratings:
                ratings[team] = INITIAL_RATING + (1 - regress) * (
                    ratings[team] - INITIAL_RATING
                )
        current_season = season_idx

        updates = {}

        for row, home, away, home_score, away_score in matches:
            for team in (home, away):
                if team not in ratings:
                    # Teams in the very first season start at the average;
                    # later newcomers (promoted teams) start lower.
                    ratings[team] = INITIAL_RATING if season_idx == 0 else promo_start

            h = ratings[home]
            a = ratings[away]

            # Store ratings BEFORE the match.
            home_elo[row] = h
            away_elo[row] = a

            expected = 1 / (1 + 10 ** ((a - (h + hfa)) / 400))
            expected_home[row] = expected

            if pd.isna(home_score) or pd.isna(away_score):
                continue  # unplayed / unknown result: no rating update

            if home_score > away_score:
                actual = 1.0
            elif home_score == away_score:
                actual = 0.5
            else:
                actual = 0.0
            actual_home[row] = actual

            multiplier = goal_multiplier(home_score - away_score) if goal_mult else 1.0
            change = k * multiplier * (actual - expected)

            # Zero-sum: home gains what away loses. Applied after the whole date.
            updates[home] = updates.get(home, 0.0) + change
            updates[away] = updates.get(away, 0.0) - change

        for team, change in updates.items():
            ratings[team] += change

    return home_elo, away_elo, expected_home, actual_home


# --------------------------------------------------
# Tuning (old seasons only)
# --------------------------------------------------

def tuning_score(df, days, params):
    _, _, expected, actual = run_elo(days, len(df), **params)

    season_idx = df["_season_idx"].to_numpy()
    last_tuning_season = df["_season_idx"].max() - N_HOLDOUT_SEASONS
    mask = (
        (season_idx >= WARMUP_SEASONS)
        & (season_idx <= last_tuning_season)
        & ~np.isnan(actual)
    )
    return float(np.mean((expected[mask] - actual[mask]) ** 2))


def tune_parameters(df, days):
    n_seasons = df["_season_idx"].max() + 1
    if n_seasons - N_HOLDOUT_SEASONS <= WARMUP_SEASONS:
        print("Not enough seasons to tune safely; using default parameters.")
        return dict(DEFAULT_PARAMS)

    plain = {"k": 20, "hfa": 0, "regress": 0.0, "promo_start": 1500, "goal_mult": False}
    plain_score = tuning_score(df, days, plain)

    best_params, best_score = None, np.inf
    keys = list(GRID.keys())

    for values in product(*(GRID[key] for key in keys)):
        params = dict(zip(keys, values))
        params["goal_mult"] = True
        score = tuning_score(df, days, params)
        if score < best_score:
            best_params, best_score = params, score

    print("\nElo tuning (mean squared error of expected score, tuning seasons only)")
    print("-" * 60)
    print(f"Plain Elo (K=20, no HFA, no regression): {plain_score:.5f}")
    print(f"Best tuned Elo: {best_score:.5f}")
    print(f"Best parameters: {best_params}")
    return best_params


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    df, _ = load_matches()
    days = build_days(df)

    params = tune_parameters(df, days) if TUNE else dict(DEFAULT_PARAMS)
    print("\nUsing Elo parameters:", params)

    home_elo, away_elo, expected_home, _ = run_elo(days, len(df), **params)

    df["home_elo"] = home_elo
    df["away_elo"] = away_elo
    df["elo_difference"] = df["home_elo"] - df["away_elo"]
    df["elo_expected_home"] = expected_home  # includes home advantage

    df = df.drop(columns=["_season_idx"])

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
                "elo_expected_home",
            ]
        ]
        .tail(10)
        .round(3)
        .to_string(index=False)
    )
    print("\nDataset shape:", df.shape)


if __name__ == "__main__":
    main()
