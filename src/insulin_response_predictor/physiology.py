"""Simple, transparent IOB and COB features for the first research milestone.

These curves are modelling approximations, not treatment calculators. Their
parameters must remain configurable and should be sensitivity-tested.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import datetime


def rapid_iob_fraction(
    age_minutes: float, *, duration_minutes: float = 300, tau_minutes: float = 90
) -> float:
    """Return the approximate fraction of a rapid dose still active."""
    if age_minutes < 0:
        raise ValueError("age_minutes cannot be negative")
    if duration_minutes <= 0 or tau_minutes <= 0:
        raise ValueError("duration_minutes and tau_minutes must be positive")
    if age_minutes >= duration_minutes:
        return 0.0
    raw = math.exp(-age_minutes / tau_minutes)
    cutoff = math.exp(-duration_minutes / tau_minutes)
    return max(0.0, min(1.0, (raw - cutoff) / (1.0 - cutoff)))


def carbs_remaining_fraction(
    age_minutes: float, *, lag_minutes: float = 10, duration_minutes: float = 240
) -> float:
    """Return unabsorbed carbohydrate fraction using a smoothstep curve."""
    if age_minutes < 0:
        raise ValueError("age_minutes cannot be negative")
    if duration_minutes <= lag_minutes:
        raise ValueError("duration_minutes must be greater than lag_minutes")
    if age_minutes <= lag_minutes:
        return 1.0
    if age_minutes >= duration_minutes:
        return 0.0
    x = (age_minutes - lag_minutes) / (duration_minutes - lag_minutes)
    absorbed = 3 * x**2 - 2 * x**3
    return max(0.0, min(1.0, 1.0 - absorbed))


def insulin_on_board(
    at: datetime,
    doses: Iterable[tuple[datetime, float]],
    *,
    duration_minutes: float = 300,
    tau_minutes: float = 90,
) -> float:
    total = 0.0
    for timestamp, units in doses:
        age = (at - timestamp).total_seconds() / 60
        if age < 0:
            continue
        total += units * rapid_iob_fraction(
            age, duration_minutes=duration_minutes, tau_minutes=tau_minutes
        )
    return total


def carbs_on_board(
    at: datetime,
    meals: Iterable[tuple[datetime, float]],
    *,
    lag_minutes: float = 10,
    duration_minutes: float = 240,
) -> float:
    total = 0.0
    for timestamp, carbs_g in meals:
        age = (at - timestamp).total_seconds() / 60
        if age < 0:
            continue
        total += carbs_g * carbs_remaining_fraction(
            age, lag_minutes=lag_minutes, duration_minutes=duration_minutes
        )
    return total
