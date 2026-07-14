"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .modeling import (
    load_bundle,
    predict_matchup,
    save_bundle,
    train_bundle,
)
from .odds import market_edge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and query a chronological UFC outcome model."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train",
        help="Train and evaluate a model.",
    )
    train_parser.add_argument("--data", type=Path, required=True)
    train_parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ufc_model.joblib"),
    )
    train_parser.add_argument("--test-size", type=float, default=0.20)
    train_parser.add_argument("--seed", type=int, default=42)

    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict one matchup.",
    )
    predict_parser.add_argument("--model", type=Path, required=True)
    predict_parser.add_argument("--red", required=True)
    predict_parser.add_argument("--blue", required=True)
    predict_parser.add_argument(
        "--date",
        default=pd.Timestamp.today().date().isoformat(),
        help="Prediction date in YYYY-MM-DD format.",
    )
    predict_parser.add_argument(
        "--title-bout",
        action="store_true",
    )
    gender_group = predict_parser.add_mutually_exclusive_group(required=True)
    gender_group.add_argument(
        "--mens-bout",
        action="store_true",
    )
    gender_group.add_argument(
        "--womens-bout",
        action="store_true",
    )
    predict_parser.add_argument("--red-odds", type=int)
    predict_parser.add_argument("--blue-odds", type=int)

    return parser


def run_train(args: argparse.Namespace) -> None:
    bundle, metrics = train_bundle(
        dataset_path=args.data,
        test_size=args.test_size,
        random_seed=args.seed,
    )
    save_bundle(bundle, args.output)

    formatted = {
        "model_path": str(args.output),
        "trained_through": bundle["trained_through"],
        "test_metrics": {
            key: round(value, 6)
            for key, value in metrics.items()
        },
    }
    print(json.dumps(formatted, indent=2))


def run_predict(args: argparse.Namespace) -> None:
    if (args.red_odds is None) != (args.blue_odds is None):
        raise ValueError(
            "Provide both --red-odds and --blue-odds, or neither."
        )

    bundle = load_bundle(args.model)
    result = predict_matchup(
        bundle=bundle,
        red_fighter=args.red,
        blue_fighter=args.blue,
        as_of=pd.Timestamp(args.date),
        title_bout=args.title_bout,
        mens_bout=args.mens_bout,
    )

    if args.red_odds is not None and args.blue_odds is not None:
        result["market"] = market_edge(
            red_model_probability=result["red_probability"],
            blue_model_probability=result["blue_probability"],
            red_odds=args.red_odds,
            blue_odds=args.blue_odds,
        )

    print(json.dumps(result, indent=2))


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        run_train(args)
    elif args.command == "predict":
        run_predict(args)


if __name__ == "__main__":
    main()
