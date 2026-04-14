from src.betting.edge import (american_to_decimal, edge, expected_value,
                              implied_prob, kelly_fraction)


def test_american_decimal():
    assert abs(american_to_decimal(+100) - 2.0) < 1e-9
    assert abs(american_to_decimal(-200) - 1.5) < 1e-9


def test_implied_and_edge():
    assert abs(implied_prob(+100) - 0.5) < 1e-9
    assert edge(0.6, +100) > 0


def test_ev_zero_at_fair():
    assert abs(expected_value(0.5, +100)) < 1e-9


def test_kelly_cap():
    assert kelly_fraction(0.99, +100) <= 0.05
    assert kelly_fraction(0.40, +100) == 0.0
