import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import poisson

from sklearn.impute import SimpleImputer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

DATA_PATH = "data/processed/matches_with_elo.csv"
OUTPUT_PATH = "reports/figures/poisson_confusion_matrix.png"

CLASSES = np.array(["H", "D", "A"])

FEATURES = [
    "home_previous_matches",
    "home_avg_goals_scored",
    "home_avg_goals_conceded",
    "away_previous_matches",
    "away_avg_goals_scored",
    "away_avg_goals_conceded",
    "home_points_per_match",
    "away_points_per_match",
    "home_avg_goal_difference",
    "away_avg_goal_difference",
    "home_last5_points",
    "away_last5_points",
    "home_last5_goal_difference",
    "away_last5_goal_difference",
    "home_elo",
    "away_elo",
    "elo_difference",
]


def load_and_split_data():
    df = pd.read_csv(DATA_PATH)

    df["match_date"] = pd.to_datetime(df["match_date"])
    df = df.sort_values("match_date").reset_index(drop=True)

    cutoff_date = pd.Timestamp("2022-10-23")

    train_df = df[df["match_date"] < cutoff_date].copy()
    test_df = df[df["match_date"] >= cutoff_date].copy()

    X_train = train_df[FEATURES]
    X_test = test_df[FEATURES]

    y_home_train = train_df["home_score"]
    y_away_train = train_df["away_score"]

    y_result_test = test_df["result"]

    print("Training matches:", len(train_df))
    print("Testing matches:", len(test_df))

    print(
        "Training period:",
        train_df["match_date"].min().date(),
        "to",
        train_df["match_date"].max().date(),
    )

    print(
        "Testing period:",
        test_df["match_date"].min().date(),
        "to",
        test_df["match_date"].max().date(),
    )

    return (
        X_train,
        X_test,
        y_home_train,
        y_away_train,
        y_result_test,
    )


def train_goal_models(X_train, y_home_train, y_away_train):
    home_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", PoissonRegressor(alpha=1.0, max_iter=3000)),
    ])

    away_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", PoissonRegressor(alpha=1.0, max_iter=3000)),
    ])

    home_model.fit(X_train, y_home_train)
    away_model.fit(X_train, y_away_train)

    return home_model, away_model


def predict_match_outcomes(home_expected, away_expected, max_goals=10):
    """
    Calculate home-win, draw, and away-win probabilities.

    Rows of score_matrix represent home goals.
    Columns of score_matrix represent away goals.
    """

    goal_counts = np.arange(max_goals + 1)

    home_goal_probabilities = poisson.pmf(
        goal_counts,
        home_expected,
    )

    away_goal_probabilities = poisson.pmf(
        goal_counts,
        away_expected,
    )

    # Rows = home goals; columns = away goals.
    score_matrix = np.outer(
        home_goal_probabilities,
        away_goal_probabilities,
    )

    # Home wins when home goals > away goals.
    home_win_probability = np.tril(
        score_matrix, k=-1
    ).sum()

    # A draw occurs when home goals = away goals.
    draw_probability = np.trace(score_matrix)

    # Away wins when away goals > home goals.
    away_win_probability = np.triu(
        score_matrix, k=1
    ).sum()

    probabilities = np.array([
        home_win_probability,
        draw_probability,
        away_win_probability,
    ])

    # Normalize to account for the small probability mass
    # outside the score range from 0 to max_goals.
    probability_sum = probabilities.sum()

    if probability_sum <= 0:
        raise ValueError("The outcome probabilities sum to zero.")

    probabilities = probabilities / probability_sum

    return probabilities

def evaluate_probability_calibration(probabilities, y_true):
    """
    Compare predicted probabilities with actual match outcomes.
    """

    print("\nProbability calibration")
    print("-----------------------")

    outcomes = ["H", "D", "A"]

    for i, outcome in enumerate(outcomes):
        predicted_probabilities = probabilities[:, i]

        actual_outcomes = (y_true.to_numpy() == outcome).astype(int)

        # Brier score: lower is better.
        brier_score = np.mean(
            (predicted_probabilities - actual_outcomes) ** 2
        )

        actual_frequency = actual_outcomes.mean()
        average_probability = predicted_probabilities.mean()

        print(f"\nOutcome: {outcome}")
        print(f"Actual frequency: {actual_frequency:.4f}")
        print(f"Average predicted probability: {average_probability:.4f}")
        print(f"Brier score: {brier_score:.4f}")


def evaluate_model(
    home_model,
    away_model,
    X_test,
    y_result_test,
):
    # Predict the expected number of goals for each team.
    home_expected = np.clip(
        home_model.predict(X_test),
        1e-6,
        None,
    )

    away_expected = np.clip(
        away_model.predict(X_test),
        1e-6,
        None,
    )

    # Display expected-goals diagnostics.
    print("\nExpected goals diagnostics")
    print("--------------------------")

    print("Home expected goals:")
    print(pd.Series(home_expected).describe())

    print("\nAway expected goals:")
    print(pd.Series(away_expected).describe())

    # Calculate the three outcome probabilities for each match.
    probabilities = np.array([
        predict_match_outcomes(home, away)
        for home, away in zip(home_expected, away_expected)
    ])
    # Analyze whether predicted draw probabilities are informative.
    print("\nDraw probability analysis")
    print("-------------------------")

    draw_analysis = pd.DataFrame({
        "predicted_draw_probability": probabilities[:, 1],
        "actual_draw": (y_result_test.to_numpy() == "D").astype(int),
    })

    bins = [0.0, 0.20, 0.25, 0.30, 0.35, 1.0]

    draw_analysis["probability_group"] = pd.cut(
        draw_analysis["predicted_draw_probability"],
        bins=bins,
        include_lowest=True,
    )

    summary = draw_analysis.groupby(
        "probability_group",
        observed=False,
    ).agg(
        matches=("actual_draw", "size"),
        average_predicted_probability=(
            "predicted_draw_probability", "mean"
        ),
        actual_draw_frequency=("actual_draw", "mean"),
    )

    print(summary.round(4))
    evaluate_probability_calibration(probabilities, y_result_test)
    # Verify that each probability row sums to approximately 1.
    assert np.allclose(
        probabilities.sum(axis=1),
        1.0,
    ), "Some outcome probabilities do not sum to 1."

    # Display probability diagnostics.
    print("\nProbability statistics")
    print("----------------------")

    for i, outcome in enumerate(["Home Win", "Draw", "Away Win"]):
        print(
            f"{outcome}: "
            f"mean={probabilities[:, i].mean():.4f}, "
            f"median={np.median(probabilities[:, i]):.4f}, "
            f"min={probabilities[:, i].min():.4f}, "
            f"max={probabilities[:, i].max():.4f}"
        )

    # Select the outcome with the highest probability.
    predictions = CLASSES[np.argmax(probabilities, axis=1)]

    print("\nPredicted outcome distribution")
    print("------------------------------")

    print(
        pd.Series(predictions)
        .value_counts(normalize=True)
        .reindex(["H", "D", "A"], fill_value=0)
        .rename(index={
            "H": "Home Win",
            "D": "Draw",
            "A": "Away Win",
        })
    )

    # Show the first five predictions and their probabilities.
    print("\nExample outcome probabilities")
    print("-----------------------------")

    for i in range(min(5, len(probabilities))):
        print(
            f"Match {i + 1}: "
            f"Home xG={home_expected[i]:.3f}, "
            f"Away xG={away_expected[i]:.3f}, "
            f"P(H)={probabilities[i, 0]:.3f}, "
            f"P(D)={probabilities[i, 1]:.3f}, "
            f"P(A)={probabilities[i, 2]:.3f}, "
            f"Prediction={predictions[i]}"
        )

    # Evaluate classification performance.
    accuracy = accuracy_score(
        y_result_test,
        predictions,
    )

    macro_f1 = f1_score(
        y_result_test,
        predictions,
        labels=["H", "D", "A"],
        average="macro",
        zero_division=0,
    )

    print("\nPoisson Model Results")
    print("---------------------")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nClassification Report")
    print("---------------------")

    print(
        classification_report(
            y_result_test,
            predictions,
            labels=["H", "D", "A"],
            target_names=["Home Win", "Draw", "Away Win"],
            zero_division=0,
        )
    )

    # Generate and save the confusion matrix.
    cm = confusion_matrix(
        y_result_test,
        predictions,
        labels=["H", "D", "A"],
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Home Win", "Draw", "Away Win"],
    )

    display.plot()
    plt.title("Poisson Model — Match Outcome Prediction")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300)
    plt.close()

    print(f"\nConfusion matrix saved to: {OUTPUT_PATH}")


def main():
    (
        X_train,
        X_test,
        y_home_train,
        y_away_train,
        y_result_test,
    ) = load_and_split_data()

    home_model, away_model = train_goal_models(
        X_train,
        y_home_train,
        y_away_train,
    )

    evaluate_model(
        home_model,
        away_model,
        X_test,
        y_result_test,
    )

    # Sanity check: the home team has higher expected goals.
    test_probabilities = predict_match_outcomes(1.5, 1.0)

    print("\nProbability function sanity check")
    print("---------------------------------")
    print(f"Home win probability: {test_probabilities[0]:.4f}")
    print(f"Draw probability: {test_probabilities[1]:.4f}")
    print(f"Away win probability: {test_probabilities[2]:.4f}")
    print(f"Sum: {test_probabilities.sum():.4f}")


if __name__ == "__main__":
    main()