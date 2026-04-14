"""Pure-function betting math: implied prob, EV, Kelly."""
from __future__ import annotations


def american_to_decimal(odds: int) -> float:
    return 1 + (odds / 100 if odds > 0 else 100 / abs(odds))


def implied_prob(odds: int) -> float:
    d = american_to_decimal(odds)
    return 1 / d


def expected_value(p: float, odds: int, stake: float = 1.0) -> float:
    """EV of a unit bet at American odds given model probability p."""
    d = american_to_decimal(odds)
    return stake * (p * (d - 1) - (1 - p))


def edge(p: float, odds: int) -> float:
    """Model prob minus market-implied prob (de-vigged naively)."""
    return p - implied_prob(odds)


def kelly_fraction(p: float, odds: int, cap: float = 0.05) -> float:
    d = american_to_decimal(odds)
    b = d - 1
    q = 1 - p
    f = (b * p - q) / b if b > 0 else 0.0
    return max(0.0, min(cap, f))
