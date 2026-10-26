"""Load and clean the multi-season Flashscore matches dataset."""

from pathlib import Path
import pandas as pd
import numpy as np


def load_and_clean_matches(csv_path):
    """Load matches.csv, remove duplicate Flashscore IDs, and prepare target/date."""
    csv_path = Path(csv_path)
    matches = pd.read_csv(csv_path)

    required = {"home_team", "away_team", "home_score", "away_score", "match_date"}
    missing = required - set(matches.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    rows_before = len(matches)

    if "flashscore_match_id" in matches.columns:
        has_id = matches["flashscore_match_id"].notna()
        matches_with_id = matches.loc[has_id].drop_duplicates(
            subset="flashscore_match_id", keep="first"
        )
        matches_without_id = matches.loc[~has_id]
        matches = pd.concat(
            [matches_with_id, matches_without_id], ignore_index=True
        )
        duplicate_rows_removed = rows_before - len(matches)
    else:
        duplicate_rows_removed = 0

    matches["match_date"] = pd.to_datetime(matches["match_date"], errors="coerce")

    # Only create a result when both final scores are available.
    valid_scores = matches["home_score"].notna() & matches["away_score"].notna()
    matches["result"] = pd.NA
    matches.loc[valid_scores, "result"] = np.select(
        [
            matches.loc[valid_scores, "home_score"]
            > matches.loc[valid_scores, "away_score"],
            matches.loc[valid_scores, "home_score"]
            == matches.loc[valid_scores, "away_score"],
        ],
        ["H", "D"],
        default="A",
    )

    print(f"Rows before cleaning: {rows_before}")
    print(f"Rows after cleaning:  {len(matches)}")
    print(f"Duplicate rows removed: {duplicate_rows_removed}")
    print(f"Rows with missing match dates: {matches['match_date'].isna().sum()}")
    print(f"Rows without a result (missing scores): {matches['result'].isna().sum()}")

    return matches.sort_values("match_date").reset_index(drop=True)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "data" / "raw" / "matches.csv"
    cleaned = load_and_clean_matches(input_path)

    output_path = project_root / "data" / "processed" / "matches_clean.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    print(f"Saved cleaned dataset to: {output_path}")
