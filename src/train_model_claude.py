"""Train and evaluate match-outcome models with walk-forward validation by season.

What this script does:
  1. Builds compact features (differences between home and away team stats).
  2. Evaluates every model on the last N_TEST_SEASONS seasons, one season at a time,
     always training only on earlier seasons (no leakage).
  3. Compares against a baseline that predicts the training H/D/A frequencies.
  4. Reports log loss and RPS (probability quality) plus accuracy.
  5. Fits final models on all data and saves them to models/.

Run from the project root:
    python src/train_model.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# Settings
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "matches_with_elo_claude.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

CLASSES = np.array(["H", "D", "A"])  # fixed column order for all probabilities
N_TEST_SEASONS = 10                  # number of walk-forward test seasons
MIN_TRAIN_SEASONS = 5                # minimum seasons needed before the first test season
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
RANDOM_STATE = 42


# --------------------------------------------------
# Data and features
# --------------------------------------------------

def build_features(df):
    """Create a small feature set. Differences keep the feature count low."""
    needed = ["home_ewm_points", "home_venue_roll10_points", "h2h_home_ppm"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing}. Re-run: python src/feature_engineering_claude.py "
            "then python src/elo_features.py"
        )

    X = pd.DataFrame(index=df.index)

    # Strength ratings
    X["elo_difference"] = df["elo_difference"]
    X["home_elo"] = df["home_elo"]
    X["away_elo"] = df["away_elo"]
    X["elo_expected_home"] = df["elo_expected_home"]  # includes home advantage

    # Recent form differences (recent windows, not the whole history)
    X["ewm_points_diff"] = df["home_ewm_points"] - df["away_ewm_points"]
    X["roll10_points_diff"] = df["home_roll10_points"] - df["away_roll10_points"]
    X["roll10_gd_diff"] = (
        df["home_roll10_goal_difference"] - df["away_roll10_goal_difference"]
    )
    X["last5_points_diff"] = df["home_last5_points"] - df["away_last5_points"]

    # Venue-specific form: home team at home vs away team away
    X["venue_points_diff"] = (
        df["home_venue_roll10_points"] - df["away_venue_roll10_points"]
    )

    # Simple expected-goals proxies: own attack combined with the opponent's defence
    X["exp_home_goals"] = (
        df["home_ewm_goals_scored"] + df["away_ewm_goals_conceded"]
    ) / 2
    X["exp_away_goals"] = (
        df["away_ewm_goals_scored"] + df["home_ewm_goals_conceded"]
    ) / 2
    X["exp_goals_diff"] = X["exp_home_goals"] - X["exp_away_goals"]

    # Context
    X["rest_diff"] = df["home_days_rest"] - df["away_days_rest"]
    X["home_promoted"] = df["home_promoted"]
    X["away_promoted"] = df["away_promoted"]
    X["matchday"] = df["matchday"]

    # Head-to-head (NaN when the teams have never met)
    X["h2h_home_ppm"] = df["h2h_home_ppm"]
    X["h2h_matches"] = df["h2h_matches"]

    # Teams with very little history give unreliable features.
    X["min_previous_matches"] = np.minimum(
        df["home_previous_matches"], df["away_previous_matches"]
    )
    return X


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["match_date"] = pd.to_datetime(df["match_date"])

    # Keep only matches with a known result.
    df = df[df["result"].isin(CLASSES)].copy()
    df = df.sort_values("match_date").reset_index(drop=True)

    # Order seasons by their first match date (not alphabetically).
    season_order = (
        df.groupby("season")["match_date"].min().sort_values().index.tolist()
    )
    print(f"Matches with a result: {len(df)}")
    print(f"Seasons: {len(season_order)} ({season_order[0]} to {season_order[-1]})")
    return df, season_order


# --------------------------------------------------
# Metrics
# --------------------------------------------------

def to_index(y):
    lookup = {c: i for i, c in enumerate(CLASSES)}
    return np.array([lookup[v] for v in y])


def log_loss_score(probs, y):
    idx = to_index(y)
    p = np.clip(probs[np.arange(len(idx)), idx], 1e-12, 1.0)
    return float(-np.mean(np.log(p)))


def rps_score(probs, y):
    """Ranked Probability Score for ordered outcomes H > D > A (lower is better)."""
    idx = to_index(y)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(idx)), idx] = 1.0
    diff = probs.cumsum(axis=1) - onehot.cumsum(axis=1)
    return float(np.mean((diff[:, :2] ** 2).sum(axis=1) / 2))


def accuracy_score_(probs, y):
    return float(np.mean(CLASSES[probs.argmax(axis=1)] == np.asarray(y)))


def summarize(probs, y):
    return {
        "log_loss": log_loss_score(probs, y),
        "rps": rps_score(probs, y),
        "accuracy": accuracy_score_(probs, y),
    }


# --------------------------------------------------
# Models
# --------------------------------------------------

def make_logreg(C):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        # No class_weight: we want honest probabilities, not forced draw predictions.
        ("clf", LogisticRegression(C=C, max_iter=5000, random_state=RANDOM_STATE)),
    ])


def make_gbm():
    # Shallow, heavily regularized: the dataset is small.
    return HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.03,
        max_iter=200,
        min_samples_leaf=40,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )


def predict_proba_ordered(model, X):
    """Return probabilities with columns ordered as CLASSES (H, D, A)."""
    proba = model.predict_proba(X)
    order = [list(model.classes_).index(c) for c in CLASSES]
    return proba[:, order]


def baseline_probs(y_train, n_rows):
    freq = np.array([(y_train == c).mean() for c in CLASSES])
    return np.tile(freq, (n_rows, 1))


def tune_C(X_train, y_train, train_seasons_col, train_seasons):
    """Pick C using the last training season as validation (still strictly past data)."""
    val_season = train_seasons[-1]
    inner_train = train_seasons_col != val_season
    inner_val = train_seasons_col == val_season

    best_C, best_loss = None, np.inf
    for C in C_GRID:
        model = make_logreg(C).fit(X_train[inner_train], y_train[inner_train])
        probs = predict_proba_ordered(model, X_train[inner_val])
        loss = log_loss_score(probs, y_train[inner_val])
        if loss < best_loss:
            best_C, best_loss = C, loss
    return best_C


# --------------------------------------------------
# Walk-forward evaluation
# --------------------------------------------------

def walk_forward(df, X, season_order):
    n_seasons = len(season_order)
    first_test = max(MIN_TRAIN_SEASONS, n_seasons - N_TEST_SEASONS)
    test_seasons = season_order[first_test:]

    y_all = df["result"].to_numpy()
    seasons_col = df["season"].to_numpy()

    model_names = ["baseline", "logreg", "gbm", "ensemble"]
    pooled_probs = {name: [] for name in model_names}
    pooled_y = []
    pooled_rows = []
    fold_rows = []

    print(f"\nWalk-forward validation on {len(test_seasons)} seasons")
    print("-" * 60)

    for test_season in test_seasons:
        i = season_order.index(test_season)
        train_seasons = season_order[:i]

        train_mask = np.isin(seasons_col, train_seasons)
        test_mask = seasons_col == test_season

        X_train, y_train = X[train_mask], y_all[train_mask]
        X_test, y_test = X[test_mask], y_all[test_mask]
        train_season_col = seasons_col[train_mask]

        best_C = tune_C(X_train, y_train, train_season_col, train_seasons)

        logreg = make_logreg(best_C).fit(X_train, y_train)
        gbm = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", make_gbm()),
        ]).fit(X_train, y_train)

        fold_probs = {
            "baseline": baseline_probs(y_train, len(y_test)),
            "logreg": predict_proba_ordered(logreg, X_test),
            "gbm": predict_proba_ordered(gbm, X_test),
        }
        fold_probs["ensemble"] = (fold_probs["logreg"] + fold_probs["gbm"]) / 2

        for name in model_names:
            pooled_probs[name].append(fold_probs[name])
            fold_rows.append({
                "season": test_season,
                "model": name,
                "matches": len(y_test),
                **summarize(fold_probs[name], y_test),
            })

        pooled_y.append(y_test)
        pooled_rows.append(df.loc[test_mask, ["match_date", "season", "home_team",
                                                "away_team", "result"]])

        print(
            f"{test_season}: n={len(y_test):3d}  C={best_C:<5}  "
            f"logloss baseline={fold_rows[-4]['log_loss']:.4f}  "
            f"logreg={fold_rows[-3]['log_loss']:.4f}  "
            f"gbm={fold_rows[-2]['log_loss']:.4f}  "
            f"ens={fold_rows[-1]['log_loss']:.4f}"
        )

    y_pooled = np.concatenate(pooled_y)
    pooled = {name: np.vstack(p) for name, p in pooled_probs.items()}

    summary = pd.DataFrame(
        {name: summarize(pooled[name], y_pooled) for name in model_names}
    ).T
    summary.index.name = "model"

    # Standard error of accuracy, so you know how much to trust the differences.
    n = len(y_pooled)
    summary["accuracy_se"] = np.sqrt(summary["accuracy"] * (1 - summary["accuracy"]) / n)

    print(f"\nPooled results over {n} test matches (lower log loss / RPS is better)")
    print("-" * 60)
    print(summary.round(4).to_string())

    best = summary["log_loss"].idxmin()
    gain = summary.loc["baseline", "log_loss"] - summary.loc[best, "log_loss"]
    print(f"\nBest by log loss: {best} "
          f"(improvement over frequency baseline: {gain:.4f})")

    return pd.DataFrame(fold_rows), pd.concat(pooled_rows, ignore_index=True), pooled


# --------------------------------------------------
# Final fit
# --------------------------------------------------

def fit_final_models(df, X, season_order):
    """Fit on all data; choose C using the latest season as validation."""
    y = df["result"].to_numpy()
    seasons_col = df["season"].to_numpy()

    best_C = tune_C(X, y, seasons_col, season_order)
    logreg = make_logreg(best_C).fit(X, y)
    gbm = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", make_gbm()),
    ]).fit(X, y)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"logreg": logreg, "gbm": gbm, "features": list(X.columns), "classes": CLASSES},
        MODELS_DIR / "outcome_models.joblib",
    )
    print(f"\nFinal models fitted on all data (C={best_C}).")
    print(f"Saved to: {MODELS_DIR / 'outcome_models.joblib'}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    df, season_order = load_data()
    X = build_features(df)

    fold_results, test_rows, pooled = walk_forward(df, X, season_order)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(REPORTS_DIR / "walk_forward_results.csv", index=False)

    predictions = test_rows.copy()
    probs = pooled["ensemble"]
    predictions["p_home"], predictions["p_draw"], predictions["p_away"] = (
        probs[:, 0], probs[:, 1], probs[:, 2]
    )
    predictions.to_csv(REPORTS_DIR / "walk_forward_predictions.csv", index=False)
    print(f"\nSaved per-season results and ensemble predictions to: {REPORTS_DIR}")

    fit_final_models(df, X, season_order)


if __name__ == "__main__":
    main()
