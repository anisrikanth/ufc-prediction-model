"""Chronological Elo and inactivity features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class FighterState:
    elo: float = 1500.0
    last_date: pd.Timestamp | None = None
    fights: int = 0


def inactivity_penalty(days_inactive: int) -> float:
    """Return the capped Elo penalty applied after 180 inactive days."""
    if days_inactive <= 180:
        return 0.0
    return min(((days_inactive - 180) / 30.0) * 1.5, 40.0)


def finish_k_factor(finish: Any) -> float:
    """Choose an Elo K-factor from the recorded result method."""
    text = str(finish).lower()
    if "ko" in text or "sub" in text:
        return 45.0
    if "split" in text or "majority" in text:
        return 25.0
    if "dec" in text:
        return 35.0
    return 30.0


def build_elo_features(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, FighterState]]:
    """Add pre-fight Elo and inactivity values without future leakage."""
    output = dataset.copy()
    tracker: dict[str, FighterState] = {}

    red_elos: list[float] = []
    blue_elos: list[float] = []
    red_rust: list[int] = []
    blue_rust: list[int] = []

    for row in output.itertuples(index=False):
        red_name = row.R_fighter
        blue_name = row.B_fighter
        fight_date = row.date

        red_state = tracker.get(red_name, FighterState())
        blue_state = tracker.get(blue_name, FighterState())

        red_days = (
            (fight_date - red_state.last_date).days
            if red_state.last_date is not None
            else 180
        )
        blue_days = (
            (fight_date - blue_state.last_date).days
            if blue_state.last_date is not None
            else 180
        )

        red_elo = red_state.elo - inactivity_penalty(red_days)
        blue_elo = blue_state.elo - inactivity_penalty(blue_days)

        red_elos.append(red_elo)
        blue_elos.append(blue_elo)
        red_rust.append(red_days)
        blue_rust.append(blue_days)

        winner = row.Winner
        if winner == "Red":
            red_score, blue_score = 1.0, 0.0
        elif winner == "Blue":
            red_score, blue_score = 0.0, 1.0
        else:
            red_score = blue_score = 0.5

        red_expected = 1.0 / (1.0 + 10.0 ** ((blue_elo - red_elo) / 400.0))
        blue_expected = 1.0 - red_expected

        finish = getattr(row, "finish", getattr(row, "Finish", ""))
        k_factor = finish_k_factor(finish)

        tracker[red_name] = FighterState(
            elo=red_elo + k_factor * (red_score - red_expected),
            last_date=fight_date,
            fights=red_state.fights + 1,
        )
        tracker[blue_name] = FighterState(
            elo=blue_elo + k_factor * (blue_score - blue_expected),
            last_date=fight_date,
            fights=blue_state.fights + 1,
        )

    output["R_elo"] = red_elos
    output["B_elo"] = blue_elos
    output["R_rust"] = red_rust
    output["B_rust"] = blue_rust
    return output, tracker


def current_elo(
    fighter_name: str,
    tracker: dict[str, FighterState],
    as_of: pd.Timestamp,
) -> float:
    """Return a fighter's Elo after applying inactivity decay as of a date."""
    state = tracker.get(fighter_name, FighterState())
    if state.last_date is None:
        return state.elo

    days_inactive = (as_of - state.last_date).days
    return state.elo - inactivity_penalty(days_inactive)


def current_rust(
    fighter_name: str,
    tracker: dict[str, FighterState],
    as_of: pd.Timestamp,
) -> int:
    """Return days since a fighter's latest recorded bout."""
    state = tracker.get(fighter_name, FighterState())
    if state.last_date is None:
        return 180
    return (as_of - state.last_date).days
