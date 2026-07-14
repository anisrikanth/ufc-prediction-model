import pytest

from ufc_model.odds import (
    american_to_decimal,
    market_edge,
    no_vig_probabilities,
)


def test_american_to_decimal() -> None:
    assert american_to_decimal(150) == pytest.approx(2.5)
    assert american_to_decimal(-200) == pytest.approx(1.5)


def test_no_vig_probabilities_sum_to_one() -> None:
    red, blue = no_vig_probabilities(-150, 130)
    assert red + blue == pytest.approx(1.0)


def test_market_edge_is_model_minus_market() -> None:
    result = market_edge(0.60, 0.40, -110, -110)
    assert result["red_edge"] == pytest.approx(0.10)
    assert result["blue_edge"] == pytest.approx(-0.10)
