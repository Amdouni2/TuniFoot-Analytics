# TuniFoot Analytics

A machine-learning project exploring Tunisian Ligue Professionnelle 1 football data and predicting match outcomes.

## Current scope

- Clean match data by removing duplicate Flashscore match identifiers when available.
- Explore goals and league standings.
- Build historical team-level features using only matches from strictly earlier dates.
- Prepare a dataset for a future home win / draw / away win classifier.

## Project structure

```text
TuniFoot/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data_cleaning.py
│   ├── exploratory_analysis.py
│   ├── feature_engineering.py
│   └── analyze_2024_2025.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Place the multi-season `matches.csv` file in `data/raw/` and the single-season file in `data/raw/` named `tunisia_ligue1_2024_2025.csv`.

Run the scripts from the project root:

```bash
python src/data_cleaning.py
python src/exploratory_analysis.py
python src/feature_engineering.py
python src/analyze_2024_2025.py
```

The processed output files are written to `data/processed/`.

## Notes

- The source dataset and its redistribution terms should be checked before publishing raw data.
- Historical features are calculated using earlier calendar dates only. Matches on the same date do not use each other's results as historical information.
- Model training and evaluation will be added after feature engineering is validated.
