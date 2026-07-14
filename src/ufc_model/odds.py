"""Market-odds utilities.

This public module reports model-versus-market differences. It intentionally
does not include bankroll sizing or proprietary wagering thresholds.
"""


def american_to_decimal(odds: int) -> float:
    """Convert non-zero American odds to decimal odds."""
    if odds == 0:
        raise ValueError("American odds cannot be zero.")
    if odds > 0:
        return (odds / 100.0) + 1.0
    return (100.0 / abs(odds)) + 1.0


def no_vig_probabilities(
    red_odds: int,
    blue_odds: int,
) -> tuple[float, float]:
    """Remove the two-way bookmaker margin by proportional normalization."""
    red_implied = 1.0 / american_to_decimal(red_odds)
    blue_implied = 1.0 / american_to_decimal(blue_odds)
    total = red_implied + blue_implied
    return red_implied / total, blue_implied / total


def market_edge(
    red_model_probability: float,
    blue_model_probability: float,
    red_odds: int,
    blue_odds: int,
) -> dict[str, float]:
    """Return model probability minus no-vig market probability."""
    red_market, blue_market = no_vig_probabilities(red_odds, blue_odds)
    return {
        "red_market_probability": red_market,
        "blue_market_probability": blue_market,
        "red_edge": red_model_probability - red_market,
        "blue_edge": blue_model_probability - blue_market,
    }
