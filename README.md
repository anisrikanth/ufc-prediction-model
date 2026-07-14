# UFC Outcome Model

A reproducible UFC bout-outcome research project built with CatBoost. The
pipeline uses chronological validation, corner-symmetrized training data,
method-adjusted Elo ratings, inactivity features, matchup statistics, and
categorical stance and weight-class features.

This repository is a cleaned public version of a personal research project.
It does **not** include the source dataset, trained model files, exact
bankroll-sizing rules, or proprietary wagering thresholds.

## What the project demonstrates

- chronological feature construction intended to avoid future-data leakage
- time-decayed and finish-adjusted fighter Elo ratings
- red-minus-blue matchup differentials
- mirrored training examples to reduce corner bias
- CatBoost handling of categorical matchup features
- holdout evaluation with accuracy, Brier score, log loss, and ROC AUC
- optional comparison of model probabilities with no-vig market probabilities

## Repository structure

```text
.
├── data/
│   └── README.md
├── artifacts/
├── src/
│   └── ufc_model/
│       ├── cli.py
│       ├── config.py
│       ├── data.py
│       ├── elo.py
│       ├── features.py
│       ├── modeling.py
│       └── odds.py
├── tests/
│   └── test_odds.py
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── requirements-dev.txt
```

## Installation

Create and activate a virtual environment, then install the package:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -e .
```

macOS or Linux:

```bash
source .venv/bin/activate
pip install -e .
```

For tests:

```bash
pip install -e ".[dev]"
pytest
```

## Dataset

The dataset is intentionally excluded from version control. Place a compatible
CSV at:

```text
data/ufc-master.csv
```

Do not republish a third-party dataset unless its license explicitly permits
redistribution. The code validates the required schema and reports missing
columns.

## Train and evaluate

```bash
ufc-model train \
  --data data/ufc-master.csv \
  --output artifacts/ufc_model.joblib
```

The newest 20% of bouts are held out by default. The command prints the model
path, latest training date, and test metrics as JSON.

Equivalent module command:

```bash
python -m ufc_model train \
  --data data/ufc-master.csv \
  --output artifacts/ufc_model.joblib
```

## Predict a matchup

```bash
ufc-model predict \
  --model artifacts/ufc_model.joblib \
  --red "Red Corner Fighter" \
  --blue "Blue Corner Fighter" \
  --date 2026-07-14 \
  --mens-bout
```

Add `--title-bout` for a championship bout.

To compare the model with a two-way market:

```bash
ufc-model predict \
  --model artifacts/ufc_model.joblib \
  --red "Red Corner Fighter" \
  --blue "Blue Corner Fighter" \
  --date 2026-07-14 \
  --mens-bout \
  --red-odds -150 \
  --blue-odds 130
```

The public version reports probability differences only. It does not make a
wager recommendation or publish bankroll-allocation logic.

## Method overview

For every historical bout, Elo and inactivity values are captured immediately
before the result updates the fighters' states. The model then uses matchup
differentials such as reach, age, striking, grappling, experience, Elo, and
inactivity. Weight class and stance matchup remain categorical.

Only the training partition is mirrored: red-minus-blue features are negated,
the stance matchup is reversed, and the target is flipped. The chronological
test partition remains untouched.

## Limitations

- Historical fighter statistics can contain missing values, inconsistent
  definitions, and source-specific errors.
- A fighter's latest row is only an approximation of their current profile.
- Rankings are neutral at prediction time unless a separate ranking input is
  added.
- Injuries, short-notice replacements, weigh-in information, judging context,
  and many stylistic details are absent.
- Backtest performance does not establish future profitability.
- Market prices change, and a measured historical edge may disappear after
  costs, limits, latency, and model decay.

## Responsible use

This project is for research and portfolio demonstration. It is not financial
advice, does not guarantee profitable predictions, and should not be used as
the sole basis for wagering decisions.

## License

No license is included by default. Add one only after deciding how you want
others to be allowed to use, modify, and redistribute the code.
