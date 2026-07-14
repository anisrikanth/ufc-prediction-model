"""Model training, evaluation, persistence, and prediction."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from .config import CATEGORICAL_FEATURES
from .data import load_dataset
from .elo import FighterState, build_elo_features
from .features import (
    build_latest_profiles,
    engineer_features,
    matchup_frame,
    symmetrize_training_data,
    training_frame,
)


def chronological_split(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split chronologically, preserving the newest observations for testing."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")

    split_index = int(len(features) * (1.0 - test_size))
    if split_index <= 0 or split_index >= len(features):
        raise ValueError("Dataset is too small for the requested split.")

    return (
        features.iloc[:split_index].copy(),
        features.iloc[split_index:].copy(),
        target.iloc[:split_index].copy(),
        target.iloc[split_index:].copy(),
    )


def build_model(random_seed: int = 42) -> CatBoostClassifier:
    """Construct the public baseline CatBoost classifier."""
    return CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=4,
        loss_function="Logloss",
        random_seed=random_seed,
        verbose=False,
    )


def evaluate(
    model: CatBoostClassifier,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, float]:
    """Compute classification and probability-quality metrics."""
    red_probabilities = model.predict_proba(features)[:, 1]
    predictions = (red_probabilities >= 0.5).astype(int)
    confidence = np.maximum(red_probabilities, 1.0 - red_probabilities)

    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "brier": float(brier_score_loss(target, red_probabilities)),
        "log_loss": float(log_loss(target, red_probabilities)),
        "roc_auc": float(roc_auc_score(target, red_probabilities)),
        "mean_confidence": float(np.mean(confidence)),
        "share_over_60": float(np.mean(confidence > 0.60)),
        "share_over_70": float(np.mean(confidence > 0.70)),
    }


def train_bundle(
    dataset_path: Path,
    test_size: float = 0.20,
    random_seed: int = 42,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Train the model and return a serializable inference bundle."""
    dataset = load_dataset(dataset_path)
    dataset, tracker = build_elo_features(dataset)
    dataset = engineer_features(dataset)

    features, target = training_frame(dataset)
    train_x, test_x, train_y, test_y = chronological_split(
        features,
        target,
        test_size,
    )

    train_x, train_y = symmetrize_training_data(train_x, train_y)
    test_x = test_x.drop(columns=["mirror_stance_matchup"])

    model = build_model(random_seed=random_seed)
    model.fit(
        train_x,
        train_y,
        cat_features=CATEGORICAL_FEATURES,
    )

    metrics = evaluate(model, test_x, test_y)
    bundle = {
        "model": model,
        "profiles": build_latest_profiles(dataset),
        "tracker": {
            name: {
                **asdict(state),
                "last_date": (
                    state.last_date.isoformat()
                    if state.last_date is not None
                    else None
                ),
            }
            for name, state in tracker.items()
        },
        "trained_through": dataset["date"].max().isoformat(),
        "metrics": metrics,
    }
    return bundle, metrics


def save_bundle(bundle: dict[str, Any], output_path: Path) -> None:
    """Save a model bundle. Only load bundles created by a trusted source."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)


def load_bundle(path: Path) -> dict[str, Any]:
    """Load a trusted model bundle and restore fighter-state objects."""
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found: {path}")

    bundle = joblib.load(path)
    bundle["tracker"] = {
        name: FighterState(
            elo=float(state["elo"]),
            last_date=(
                pd.Timestamp(state["last_date"])
                if state["last_date"] is not None
                else None
            ),
            fights=int(state["fights"]),
        )
        for name, state in bundle["tracker"].items()
    }
    return bundle


def predict_matchup(
    bundle: dict[str, Any],
    red_fighter: str,
    blue_fighter: str,
    as_of: pd.Timestamp,
    title_bout: bool,
    mens_bout: bool,
) -> dict[str, Any]:
    """Predict a matchup from a previously trained bundle."""
    profiles = bundle["profiles"]
    red_profile = profiles.get(red_fighter.strip().lower())
    blue_profile = profiles.get(blue_fighter.strip().lower())

    missing = []
    if red_profile is None:
        missing.append(red_fighter)
    if blue_profile is None:
        missing.append(blue_fighter)
    if missing:
        raise ValueError(
            "No profile found for: " + ", ".join(missing)
        )

    frame = matchup_frame(
        red_profile=red_profile,
        blue_profile=blue_profile,
        tracker=bundle["tracker"],
        as_of=as_of,
        title_bout=title_bout,
        mens_bout=mens_bout,
    )

    red_probability = float(bundle["model"].predict_proba(frame)[0, 1])
    blue_probability = 1.0 - red_probability
    predicted_winner = (
        red_profile["official_name"]
        if red_probability >= blue_probability
        else blue_profile["official_name"]
    )

    return {
        "red_fighter": red_profile["official_name"],
        "blue_fighter": blue_profile["official_name"],
        "red_probability": red_probability,
        "blue_probability": blue_probability,
        "predicted_winner": predicted_winner,
        "as_of": as_of.isoformat(),
        "red_dataset_fights": bundle["tracker"].get(
            red_profile["official_name"],
            FighterState(),
        ).fights,
        "blue_dataset_fights": bundle["tracker"].get(
            blue_profile["official_name"],
            FighterState(),
        ).fights,
    }
