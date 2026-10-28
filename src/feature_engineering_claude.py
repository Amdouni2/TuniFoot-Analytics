"""Build leakage-aware historical team features for match-outcome prediction.

Every feature for a match uses only matches played on strictly earlier dates.

Feature groups (each is created for both teams with a `home_` / `away_` prefix):
  - legacy cumulative averages (kept so poisson_model.py and older scripts still work)
  - rolling form over the last 10 matches (all venues)
  - exponentially weighted form (half-life of 10 matches)
  - venue-specific form (home team's recent HOME matches, away team's recent AWAY matches)
  - days of rest since the previous match
  - promoted flag (team was not in the league the previous season)
Plus match-level head-to-head features over the last 5 meetings.

Pipeline order:
    python src/feature_engineering.py
    python src/elo_features.py
    python src/train_model.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

from data_cleaning import load_and_clean_matches

ROLL_WINDOW = 10      # matches in the rolling form window
FORM_WINDOW = 5       # matches in the "recent form" window
EWM_HALFLIFE = 10     # matches for exponential weighting
MIN_PERIODS = 3       # NaN until a team has at least this many earlier matches
H2H_WINDOW = 5        # previous meetings used for head-to-head
MAX_REST_DAYS = 30    # cap long breaks (off-season, suspensions)

TEAM_FEATURES = [
    # legacy cumulative features (whole history)
    "previous_matches",
    "avg_goals_scored",
    "avg_goals_conceded",
    "points_per_match",
    "avg_goal_difference",
    # recent form (mean over last 5 matches, NaN if fewer than MIN_PERIODS)
    "last5_points",
    "last5_goal_difference",
    # rolling window, all venues
    "roll10_points",
    "roll10_goals_scored",
    "roll10_goals_conceded",
    "roll10_goal_difference",
    # exponentially weighted
    "ewm_points",
    "ewm_goals_scored",
    "ewm_goals_conceded",
    # same-venue form
    "venue_roll10_points",
    "venue_roll10_goals_scored",
    "venue_roll10_goals_conceded",
    # context
    "days_rest",
    "promoted",
]

H2H_FEATURES = ["h2h_matches", "h2h_home_ppm", "h2h_home_gd"]


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def _roll_mean(frame, keys, col, window):
    """Mean of the previous `window` rows within each group (current row excluded)."""
    return frame.groupby(keys, sort=False)[col].transform(
        lambda s: s.shift(1).rolling(window=window, min_periods=MIN_PERIODS).mean()
    )


def _ewm_mean(frame, keys, col, halflife):
    """Exponentially weighted mean of previous rows within each group."""
    return frame.groupby(keys, sort=False)[col].transform(
        lambda s: s.shift(1).ewm(halflife=halflife, min_periods=MIN_PERIODS).mean()
    )


def _promoted_lookup(matches):
    """Return a set of (team, season) pairs for teams absent from the previous season."""
    if "season" not in matches.columns:
        return set()

    df = matches.dropna(subset=["season", "match_date", "home_team", "away_team"])
    if df.empty:
        return set()

    season_order = (
        df.groupby("season")["match_date"].min().sort_values().index.tolist()
    )
    teams_by_season = {
        s: set(df.loc[df["season"] == s, "home_team"])
        | set(df.loc[df["season"] == s, "away_team"])
        for s in season_order
    }

    promoted = set()
    for previous, current in zip(season_order[:-1], season_order[1:]):
        for team in teams_by_season[current] - teams_by_season[previous]:
            promoted.add((team, current))
    return promoted


def _build_team_matches(usable):
    """Turn each match into two rows: one per team."""
    cols = ["_row_id", "match_date", "season"] if "season" in usable.columns else [
        "_row_id", "match_date"
    ]

    home = usable[cols + ["home_team", "home_score", "away_score"]].copy()
    home.columns = cols + ["team", "goals_scored", "goals_conceded"]
    home["team_role"] = "home"

    away = usable[cols + ["away_team", "away_score", "home_score"]].copy()
    away.columns = cols + ["team", "goals_scored", "goals_conceded"]
    away["team_role"] = "away"

    tm = pd.concat([home, away], ignore_index=True)
    tm["points"] = np.where(
        tm["goals_scored"] > tm["goals_conceded"], 3,
        np.where(tm["goals_scored"] == tm["goals_conceded"], 1, 0),
    )
    tm["goal_difference"] = tm["goals_scored"] - tm["goals_conceded"]
    return tm


def _add_team_history(tm, promoted):
    """Add all per-team historical features to the team-match table."""
    tm = tm.sort_values(
        ["team", "match_date", "_row_id"], kind="mergesort"
    ).reset_index(drop=True)

    # --- legacy cumulative features (all earlier matches) ---
    tm["previous_matches"] = tm.groupby("team", sort=False).cumcount()
    denom = tm["previous_matches"].replace(0, np.nan)

    for source, target in [
        ("goals_scored", "avg_goals_scored"),
        ("goals_conceded", "avg_goals_conceded"),
        ("points", "points_per_match"),
        ("goal_difference", "avg_goal_difference"),
    ]:
        running = tm.groupby("team", sort=False)[source].cumsum() - tm[source]
        tm[target] = running / denom

    # --- recent form ---
    tm["last5_points"] = _roll_mean(tm, "team", "points", FORM_WINDOW)
    tm["last5_goal_difference"] = _roll_mean(tm, "team", "goal_difference", FORM_WINDOW)

    # --- rolling window over all venues ---
    tm["roll10_points"] = _roll_mean(tm, "team", "points", ROLL_WINDOW)
    tm["roll10_goals_scored"] = _roll_mean(tm, "team", "goals_scored", ROLL_WINDOW)
    tm["roll10_goals_conceded"] = _roll_mean(tm, "team", "goals_conceded", ROLL_WINDOW)
    tm["roll10_goal_difference"] = _roll_mean(tm, "team", "goal_difference", ROLL_WINDOW)

    # --- exponentially weighted form ---
    tm["ewm_points"] = _ewm_mean(tm, "team", "points", EWM_HALFLIFE)
    tm["ewm_goals_scored"] = _ewm_mean(tm, "team", "goals_scored", EWM_HALFLIFE)
    tm["ewm_goals_conceded"] = _ewm_mean(tm, "team", "goals_conceded", EWM_HALFLIFE)

    # --- same-venue form (home team's HOME matches, away team's AWAY matches) ---
    venue_keys = ["team", "team_role"]
    tm["venue_roll10_points"] = _roll_mean(tm, venue_keys, "points", ROLL_WINDOW)
    tm["venue_roll10_goals_scored"] = _roll_mean(
        tm, venue_keys, "goals_scored", ROLL_WINDOW
    )
    tm["venue_roll10_goals_conceded"] = _roll_mean(
        tm, venue_keys, "goals_conceded", ROLL_WINDOW
    )

    # --- rest days since the team's previous match ---
    tm["days_rest"] = (
        tm.groupby("team", sort=False)["match_date"].diff().dt.days.clip(upper=MAX_REST_DAYS)
    )

    # --- promoted flag (known before the season starts, so no leakage) ---
    if "season" in tm.columns:
        tm["promoted"] = [
            int((team, season) in promoted)
            for team, season in zip(tm["team"], tm["season"])
        ]
    else:
        tm["promoted"] = 0

    # --- same-date safety ---
    # If a team has more than one match on a date (data error / rescheduling),
    # give all of them the features of the earliest one, which only saw earlier dates.
    duplicated = tm.duplicated(["team", "match_date"], keep=False)
    if duplicated.any():
        fixed = (
            tm.loc[duplicated]
            .groupby(["team", "match_date"], sort=False)[TEAM_FEATURES]
            .transform(lambda s: s.iloc[0])
        )
        tm.loc[duplicated, TEAM_FEATURES] = fixed

    return tm


def _build_h2h(usable):
    """Head-to-head features from the last H2H_WINDOW meetings, strictly before the date."""
    usable = usable.sort_values(["match_date", "_row_id"], kind="mergesort")

    history = {}
    rows = []

    for _, day in usable.groupby("match_date", sort=True):
        updates = []

        for row_id, home, away, home_score, away_score in zip(
            day["_row_id"],
            day["home_team"],
            day["away_team"],
            day["home_score"],
            day["away_score"],
        ):
            key = frozenset((home, away))
            past = history.get(key, [])[-H2H_WINDOW:]

            points, goal_diffs = [], []
            for past_home, past_home_score, past_away_score in past:
                # Express each past meeting from the current home team's perspective.
                if past_home == home:
                    goals_for, goals_against = past_home_score, past_away_score
                else:
                    goals_for, goals_against = past_away_score, past_home_score
                goal_diffs.append(goals_for - goals_against)
                points.append(
                    3 if goals_for > goals_against else 1 if goals_for == goals_against else 0
                )

            rows.append({
                "_row_id": row_id,
                "h2h_matches": len(past),
                "h2h_home_ppm": np.mean(points) if points else np.nan,
                "h2h_home_gd": np.mean(goal_diffs) if goal_diffs else np.nan,
            })
            updates.append((key, (home, home_score, away_score)))

        # Update only after the whole date is processed (no same-day leakage).
        for key, item in updates:
            history.setdefault(key, []).append(item)

    return pd.DataFrame(rows, columns=["_row_id"] + H2H_FEATURES)


# --------------------------------------------------
# Main feature builder
# --------------------------------------------------

def add_historical_team_features(matches):
    """Add historical features for both teams plus head-to-head features."""
    required = {"home_team", "away_team", "home_score", "away_score", "match_date"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    result = matches.copy().reset_index(drop=True)
    result["_row_id"] = range(len(result))
    result["match_date"] = pd.to_datetime(result["match_date"], errors="coerce")

    promoted = _promoted_lookup(result)

    usable = result.dropna(
        subset=["match_date", "home_team", "away_team", "home_score", "away_score"]
    ).copy()

    team_matches = _build_team_matches(usable)
    team_matches = _add_team_history(team_matches, promoted)

    for role in ["home", "away"]:
        role_features = team_matches.loc[
            team_matches["team_role"] == role, ["_row_id"] + TEAM_FEATURES
        ].rename(columns={c: f"{role}_{c}" for c in TEAM_FEATURES})
        result = result.merge(
            role_features, on="_row_id", how="left", validate="one_to_one"
        )

    h2h = _build_h2h(usable)
    result = result.merge(h2h, on="_row_id", how="left", validate="one_to_one")

    return result.drop(columns=["_row_id"])


def check_no_leakage(featured):
    """Basic sanity checks. Raises AssertionError if something looks wrong."""
    for role in ["home", "away"]:
        first_matches = featured[featured[f"{role}_previous_matches"] == 0]
        assert first_matches[f"{role}_ewm_points"].isna().all(), (
            f"{role}: ewm features exist for teams with no history"
        )
        assert first_matches[f"{role}_roll10_points"].isna().all(), (
            f"{role}: rolling features exist for teams with no history"
        )
        assert first_matches[f"{role}_days_rest"].isna().all(), (
            f"{role}: rest days exist for teams with no history"
        )

    no_meetings = featured[featured["h2h_matches"] == 0]
    assert no_meetings["h2h_home_ppm"].isna().all(), "h2h features without meetings"
    print("Leakage sanity checks passed.")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "data" / "raw" / "matches.csv"

    matches = load_and_clean_matches(input_path)
    featured_matches = add_historical_team_features(matches)
    check_no_leakage(featured_matches)

    output_path = project_root / "data" / "processed" / "matches_with_features_claude.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    featured_matches.to_csv(output_path, index=False)

    new_columns = [f"{role}_{c}" for role in ["home", "away"] for c in TEAM_FEATURES]
    new_columns += H2H_FEATURES

    print("\nSample of generated features:")
    print(
        featured_matches[
            ["match_date", "home_team", "away_team", "result"]
            + [
                "home_roll10_points",
                "away_roll10_points",
                "home_ewm_points",
                "away_ewm_points",
                "home_venue_roll10_points",
                "away_venue_roll10_points",
                "home_days_rest",
                "away_days_rest",
                "h2h_matches",
                "h2h_home_ppm",
            ]
        ]
        .tail(10)
        .round(2)
        .to_string(index=False)
    )

    print("\nDataset shape:", featured_matches.shape)
    print("\nMissing values in engineered features:")
    print(featured_matches[new_columns].isnull().sum().to_string())
    print(f"\nSaved feature dataset to: {output_path}")
