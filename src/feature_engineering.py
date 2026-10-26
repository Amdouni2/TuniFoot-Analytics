"""Build leakage-aware historical team features for match-outcome prediction."""

from pathlib import Path
import pandas as pd

from data_cleaning import load_and_clean_matches


def add_historical_team_features(matches):
    """
    Add each team's average goals scored/conceded before the match date.

    Matches on the same calendar date are grouped together. Their results are
    therefore not used as historical information for any other match on that date.
    """
    required = {
        "home_team", "away_team", "home_score", "away_score", "match_date"
    }
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    result = matches.copy().reset_index(drop=True)
    result["_row_id"] = range(len(result))
    result["match_date"] = pd.to_datetime(result["match_date"], errors="coerce")

    # Keep only rows that have teams, a date, and final scores for feature creation.
    usable = result.dropna(
        subset=["match_date", "home_team", "away_team", "home_score", "away_score"]
    ).copy()

    home_view = usable[
        ["_row_id", "match_date", "home_team", "home_score", "away_score"]
    ].copy()
    home_view.columns = [
        "_row_id", "match_date", "team", "goals_scored", "goals_conceded"
    ]
    home_view["team_role"] = "home"

    away_view = usable[
        ["_row_id", "match_date", "away_team", "away_score", "home_score"]
    ].copy()
    away_view.columns = [
        "_row_id", "match_date", "team", "goals_scored", "goals_conceded"
    ]
    away_view["team_role"] = "away"

    team_matches = pd.concat([home_view, away_view], ignore_index=True)

    # Aggregate by team and date to prevent same-day matches leaking into features.
    daily_stats = team_matches.groupby(["team", "match_date"], as_index=False).agg(
        daily_goals_scored=("goals_scored", "sum"),
        daily_goals_conceded=("goals_conceded", "sum"),
        matches_played=("goals_scored", "count"),
    )
    daily_stats = daily_stats.sort_values(["team", "match_date"]).reset_index(drop=True)

    grouped = daily_stats.groupby("team", sort=False)
    daily_stats["cumulative_goals"] = grouped["daily_goals_scored"].cumsum()
    daily_stats["cumulative_conceded"] = grouped["daily_goals_conceded"].cumsum()
    daily_stats["cumulative_matches"] = grouped["matches_played"].cumsum()

    daily_stats["previous_goals"] = (
        daily_stats.groupby("team")["cumulative_goals"].shift(1).fillna(0)
    )
    daily_stats["previous_conceded_goals"] = (
        daily_stats.groupby("team")["cumulative_conceded"].shift(1).fillna(0)
    )
    daily_stats["previous_matches"] = (
        daily_stats.groupby("team")["cumulative_matches"].shift(1).fillna(0)
    )

    # Average is NaN for teams with no earlier recorded matches.
    daily_stats["avg_goals_scored"] = (
        daily_stats["previous_goals"]
        / daily_stats["previous_matches"].replace(0, pd.NA)
    )
    daily_stats["avg_goals_conceded"] = (
        daily_stats["previous_conceded_goals"]
        / daily_stats["previous_matches"].replace(0, pd.NA)
    )

    history_columns = [
        "team", "match_date", "previous_matches",
        "avg_goals_scored", "avg_goals_conceded",
    ]
    team_matches = team_matches.merge(
        daily_stats[history_columns],
        on=["team", "match_date"],
        how="left",
        validate="many_to_one",
    )

    home_features = team_matches.loc[
        team_matches["team_role"] == "home",
        ["_row_id", "previous_matches", "avg_goals_scored", "avg_goals_conceded"],
    ].rename(columns={
        "previous_matches": "home_previous_matches",
        "avg_goals_scored": "home_avg_goals_scored",
        "avg_goals_conceded": "home_avg_goals_conceded",
    })

    away_features = team_matches.loc[
        team_matches["team_role"] == "away",
        ["_row_id", "previous_matches", "avg_goals_scored", "avg_goals_conceded"],
    ].rename(columns={
        "previous_matches": "away_previous_matches",
        "avg_goals_scored": "away_avg_goals_scored",
        "avg_goals_conceded": "away_avg_goals_conceded",
    })

    result = result.merge(home_features, on="_row_id", how="left", validate="one_to_one")
    result = result.merge(away_features, on="_row_id", how="left", validate="one_to_one")
    result = result.drop(columns=["_row_id"])

    return result


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "data" / "raw" / "matches.csv"
    matches = load_and_clean_matches(input_path)
    featured_matches = add_historical_team_features(matches)

    output_path = project_root / "data" / "processed" / "matches_with_features.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    featured_matches.to_csv(output_path, index=False)

    feature_columns = [
        "match_date", "home_team", "away_team", "result",
        "home_previous_matches", "home_avg_goals_scored",
        "home_avg_goals_conceded", "away_previous_matches",
        "away_avg_goals_scored", "away_avg_goals_conceded",
    ]
    print(featured_matches[feature_columns].head(10).to_string(index=False))
    print(f"\nSaved feature dataset to: {output_path}")
