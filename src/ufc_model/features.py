"""Feature engineering for training and inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS
from .elo import FighterState, current_elo, current_rust


def parse_rank(value: Any) -> int:
    """Encode the corner with the better ranking."""
    if pd.isna(value) or value == "neither":
        return 0
    if value == "Red":
        return 1
    if value == "Blue":
        return -1
    return 0


def engineer_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Create matchup interactions and red-minus-blue differentials."""
    output = dataset.copy()

    output["stance_matchup"] = (
        output["R_Stance"] + "_vs_" + output["B_Stance"]
    )
    output["mirror_stance_matchup"] = (
        output["B_Stance"] + "_vs_" + output["R_Stance"]
    )

    if "better_rank" in output.columns:
        output["better_rank_code"] = output["better_rank"].apply(parse_rank)
    else:
        output["better_rank_code"] = 0

    pairs = {
        "reach_diff": ("R_Reach_cms", "B_Reach_cms"),
        "age_diff": ("R_age", "B_age"),
        "sig_str_diff": ("R_avg_SIG_STR_landed", "B_avg_SIG_STR_landed"),
        "sig_str_pct_diff": ("R_avg_SIG_STR_pct", "B_avg_SIG_STR_pct"),
        "td_diff": ("R_avg_TD_landed", "B_avg_TD_landed"),
        "td_pct_diff": ("R_avg_TD_pct", "B_avg_TD_pct"),
        "sub_diff": ("R_avg_SUB_ATT", "B_avg_SUB_ATT"),
        "rounds_fought_diff": (
            "R_total_rounds_fought",
            "B_total_rounds_fought",
        ),
        "title_bouts_diff": ("R_total_title_bouts", "B_total_title_bouts"),
        "ko_win_diff": ("R_win_by_KO/TKO", "B_win_by_KO/TKO"),
        "sub_win_diff": (
            "R_win_by_Submission",
            "B_win_by_Submission",
        ),
        "dec_win_diff": (
            "R_win_by_Decision_Unanimous",
            "B_win_by_Decision_Unanimous",
        ),
        "elo_diff": ("R_elo", "B_elo"),
        "rust_diff": ("R_rust", "B_rust"),
    }

    for feature_name, (red_column, blue_column) in pairs.items():
        output[feature_name] = output[red_column] - output[blue_column]

    red_striking_pressure = (
        output["R_avg_SIG_STR_landed"] * output["R_avg_SIG_STR_pct"]
    )
    blue_striking_pressure = (
        output["B_avg_SIG_STR_landed"] * output["B_avg_SIG_STR_pct"]
    )
    output["striking_pressure_diff"] = (
        red_striking_pressure - blue_striking_pressure
    )

    red_grappling_pressure = (
        output["R_avg_TD_landed"] + output["R_avg_SUB_ATT"]
    )
    blue_grappling_pressure = (
        output["B_avg_TD_landed"] + output["B_avg_SUB_ATT"]
    )
    output["grappling_pressure_diff"] = (
        red_grappling_pressure - blue_grappling_pressure
    )

    return output


def training_frame(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return model features and a binary red-corner win target."""
    clean = dataset[dataset["Winner"].isin(["Red", "Blue"])].copy()
    target = clean["Winner"].map({"Red": 1, "Blue": 0}).astype(int)
    columns = FEATURE_COLUMNS + ["mirror_stance_matchup"]
    return clean[columns].copy(), target


def symmetrize_training_data(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Mirror corner-dependent features to reduce red-corner bias."""
    original = features.copy()
    mirrored = features.copy()

    differential_columns = [
        column for column in FEATURE_COLUMNS if column.endswith("_diff")
    ]
    mirrored[differential_columns] = -original[differential_columns]
    mirrored["better_rank_code"] = -original["better_rank_code"]
    mirrored["stance_matchup"] = original["mirror_stance_matchup"]

    original = original.drop(columns=["mirror_stance_matchup"])
    mirrored = mirrored.drop(columns=["mirror_stance_matchup"])

    combined_features = pd.concat(
        [original, mirrored],
        ignore_index=True,
    )
    combined_target = pd.concat(
        [target.reset_index(drop=True), 1 - target.reset_index(drop=True)],
        ignore_index=True,
    )
    return combined_features, combined_target


def build_latest_profiles(dataset: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Build one inference profile per fighter from their latest row."""
    profiles: dict[str, dict[str, Any]] = {}

    for _, row in dataset.iterrows():
        red_profile = _profile_from_row(row, "R")
        blue_profile = _profile_from_row(row, "B")
        profiles[red_profile["official_name"].lower()] = red_profile
        profiles[blue_profile["official_name"].lower()] = blue_profile

    return profiles


def _profile_from_row(row: pd.Series, corner: str) -> dict[str, Any]:
    return {
        "official_name": row[f"{corner}_fighter"],
        "weight_class": str(row["weight_class"]),
        "stance": str(row.get(f"{corner}_Stance", "Unknown")),
        "reach": row[f"{corner}_Reach_cms"],
        "age": row[f"{corner}_age"],
        "sig_str": row[f"{corner}_avg_SIG_STR_landed"],
        "sig_str_pct": row[f"{corner}_avg_SIG_STR_pct"],
        "td": row[f"{corner}_avg_TD_landed"],
        "td_pct": row[f"{corner}_avg_TD_pct"],
        "sub": row[f"{corner}_avg_SUB_ATT"],
        "rounds": row[f"{corner}_total_rounds_fought"],
        "title_bouts": row[f"{corner}_total_title_bouts"],
        "ko": row[f"{corner}_win_by_KO/TKO"],
        "sub_win": row[f"{corner}_win_by_Submission"],
        "dec": row[f"{corner}_win_by_Decision_Unanimous"],
    }


def matchup_frame(
    red_profile: dict[str, Any],
    blue_profile: dict[str, Any],
    tracker: dict[str, FighterState],
    as_of: pd.Timestamp,
    title_bout: bool,
    mens_bout: bool,
) -> pd.DataFrame:
    """Create a one-row model input for a future matchup."""
    red_name = red_profile["official_name"]
    blue_name = blue_profile["official_name"]

    red_striking_pressure = (
        red_profile["sig_str"] * red_profile["sig_str_pct"]
    )
    blue_striking_pressure = (
        blue_profile["sig_str"] * blue_profile["sig_str_pct"]
    )
    red_grappling_pressure = red_profile["td"] + red_profile["sub"]
    blue_grappling_pressure = blue_profile["td"] + blue_profile["sub"]

    values = {
        "title_bout": int(title_bout),
        "gender": int(mens_bout),
        "better_rank_code": 0,
        "reach_diff": red_profile["reach"] - blue_profile["reach"],
        "age_diff": red_profile["age"] - blue_profile["age"],
        "sig_str_diff": red_profile["sig_str"] - blue_profile["sig_str"],
        "sig_str_pct_diff": (
            red_profile["sig_str_pct"] - blue_profile["sig_str_pct"]
        ),
        "td_diff": red_profile["td"] - blue_profile["td"],
        "td_pct_diff": red_profile["td_pct"] - blue_profile["td_pct"],
        "sub_diff": red_profile["sub"] - blue_profile["sub"],
        "rounds_fought_diff": (
            red_profile["rounds"] - blue_profile["rounds"]
        ),
        "title_bouts_diff": (
            red_profile["title_bouts"] - blue_profile["title_bouts"]
        ),
        "ko_win_diff": red_profile["ko"] - blue_profile["ko"],
        "sub_win_diff": red_profile["sub_win"] - blue_profile["sub_win"],
        "dec_win_diff": red_profile["dec"] - blue_profile["dec"],
        "elo_diff": (
            current_elo(red_name, tracker, as_of)
            - current_elo(blue_name, tracker, as_of)
        ),
        "rust_diff": (
            current_rust(red_name, tracker, as_of)
            - current_rust(blue_name, tracker, as_of)
        ),
        "striking_pressure_diff": (
            red_striking_pressure - blue_striking_pressure
        ),
        "grappling_pressure_diff": (
            red_grappling_pressure - blue_grappling_pressure
        ),
        "weight_class": red_profile["weight_class"],
        "stance_matchup": (
            f"{red_profile['stance']}_vs_{blue_profile['stance']}"
        ),
    }

    frame = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    numeric_columns = frame.select_dtypes(include=[np.number]).columns
    frame[numeric_columns] = frame[numeric_columns].fillna(0.0)
    return frame
