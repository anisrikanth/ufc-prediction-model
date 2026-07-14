"""Dataset loading and validation."""

from pathlib import Path

import pandas as pd

from .config import REQUIRED_COLUMNS


def load_dataset(path: Path) -> pd.DataFrame:
    """Load, validate, and chronologically sort a UFC bout dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataset = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS.difference(dataset.columns))
    if missing:
        raise ValueError(
            "Dataset is missing required columns: " + ", ".join(missing)
        )

    dataset = dataset.copy()
    dataset["date"] = pd.to_datetime(dataset["date"], errors="raise")
    dataset = dataset.sort_values("date").reset_index(drop=True)

    dataset["title_bout"] = dataset["title_bout"].astype(int)
    dataset["gender"] = (
        dataset["gender"]
        .astype(str)
        .str.upper()
        .eq("MALE")
        .astype(int)
    )

    for column in ("weight_class", "R_Stance", "B_Stance"):
        dataset[column] = dataset[column].fillna("Unknown").astype(str)

    return dataset
