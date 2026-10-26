"""Basic analysis of the single-season 2024–2025 CSV."""

from pathlib import Path
import pandas as pd


def analyze_season(csv_path):
    df = pd.read_csv(csv_path, parse_dates=["date"])

    required = {"home_team", "away_team", "home_goals", "away_goals"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    total_home_goals = df["home_goals"].sum()
    total_away_goals = df["away_goals"].sum()
    total_goals = total_home_goals + total_away_goals

    print("Total home goals scored:", total_home_goals)
    print("Total away goals scored:", total_away_goals)
    print("Average home goals scored:", df["home_goals"].mean())
    print("Average away goals scored:", df["away_goals"].mean())
    print("Total goals scored:", total_goals)
    print("Average total goals per match:", (df["home_goals"] + df["away_goals"]).mean())

    team = df["home_team"].dropna().unique()[0]
    team_matches = df[(df["home_team"] == team) | (df["away_team"] == team)]

    goals_scored = (
        team_matches.loc[team_matches["home_team"] == team, "home_goals"].sum()
        + team_matches.loc[team_matches["away_team"] == team, "away_goals"].sum()
    )

    print(f"\nExample team: {team}")
    print(f"Matches played by {team}: {len(team_matches)}")
    print(f"Goals scored by {team}: {goals_scored}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / "raw" / "tunisia_ligue1_2024_2025.csv"
    analyze_season(csv_path)
