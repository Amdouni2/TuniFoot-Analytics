"""Exploratory statistics and league standings for the multi-season dataset."""

from pathlib import Path
import pandas as pd

from data_cleaning import load_and_clean_matches


def print_match_summary(matches):
    """Print high-level statistics for matches with valid final scores."""
    played = matches.dropna(subset=["home_score", "away_score"]).copy()
    if played.empty:
        print("No matches with valid scores were found.")
        return

    total_home_goals = played["home_score"].sum()
    total_away_goals = played["away_score"].sum()

    print("\nMATCH SUMMARY")
    print(f"Matches with scores: {len(played)}")
    print(f"Total home goals: {total_home_goals}")
    print(f"Total away goals: {total_away_goals}")
    print(f"Average home goals per match: {played['home_score'].mean():.3f}")
    print(f"Average away goals per match: {played['away_score'].mean():.3f}")
    print(
        "Average total goals per match: "
        f"{(played['home_score'] + played['away_score']).mean():.3f}"
    )


def calculate_standings(matches, season):
    """Calculate a league table for one season using matches with final scores."""
    required = {"season", "home_team", "away_team", "home_score", "away_score"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    season_matches = matches.loc[
        (matches["season"] == season)
        & matches["home_score"].notna()
        & matches["away_score"].notna()
    ].copy()

    if season_matches.empty:
        raise ValueError(f"No completed matches with scores found for season {season!r}.")

    home_stats = pd.DataFrame({
        "team": season_matches["home_team"],
        "played": 1,
        "wins": (season_matches["home_score"] > season_matches["away_score"]).astype(int),
        "draws": (season_matches["home_score"] == season_matches["away_score"]).astype(int),
        "losses": (season_matches["home_score"] < season_matches["away_score"]).astype(int),
        "goals_for": season_matches["home_score"],
        "goals_against": season_matches["away_score"],
    })

    away_stats = pd.DataFrame({
        "team": season_matches["away_team"],
        "played": 1,
        "wins": (season_matches["away_score"] > season_matches["home_score"]).astype(int),
        "draws": (season_matches["away_score"] == season_matches["home_score"]).astype(int),
        "losses": (season_matches["away_score"] < season_matches["home_score"]).astype(int),
        "goals_for": season_matches["away_score"],
        "goals_against": season_matches["home_score"],
    })

    standings = pd.concat([home_stats, away_stats], ignore_index=True)
    standings = standings.groupby("team", as_index=True).sum(numeric_only=True)
    standings["goal_difference"] = standings["goals_for"] - standings["goals_against"]
    standings["points"] = standings["wins"] * 3 + standings["draws"]
    standings["points_per_match"] = standings["points"] / standings["played"]
    standings["goals_per_match"] = standings["goals_for"] / standings["played"]

    standings = standings.sort_values(
        ["points", "goal_difference", "goals_for"],
        ascending=[False, False, False],
    )
    standings.insert(0, "position", range(1, len(standings) + 1))
    return standings


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "data" / "raw" / "matches.csv"
    matches = load_and_clean_matches(input_path)

    print_match_summary(matches)

    season_to_analyze = "2025/2026"
    try:
        table = calculate_standings(matches, season_to_analyze)
        print(f"\nLEAGUE TABLE — {season_to_analyze}")
        print(table.to_string())
        print(f"\nBest attack: {table['goals_for'].idxmax()}")
        print(f"Best defence: {table['goals_against'].idxmin()}")
    except ValueError as exc:
        print(f"\nCould not calculate standings: {exc}")
