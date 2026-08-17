"""Methodology configuration and validation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, Mapping


DEFAULT_CONFIG: Dict[str, Any] = {
    "methodology": {
        "category_weights": {
            "fundamentals": 0.30,
            "valuation": 0.25,
            "momentum": 0.30,
            "risk": 0.15,
        },
        "factors": {
            "fundamentals": {
                "roe": {"weight": 0.35, "direction": "higher"},
                "operating_margin": {"weight": 0.25, "direction": "higher"},
                "revenue_growth": {"weight": 0.25, "direction": "higher"},
                "debt_to_equity": {"weight": 0.15, "direction": "lower"},
            },
            "valuation": {
                "pe": {"weight": 0.35, "direction": "lower", "positive_only": True},
                "pb": {"weight": 0.20, "direction": "lower", "positive_only": True},
                "ev_ebitda": {"weight": 0.30, "direction": "lower", "positive_only": True},
                "dividend_yield": {"weight": 0.15, "direction": "higher"},
            },
            "momentum": {
                "return_6m": {"weight": 0.50, "direction": "higher"},
                "return_12_1m": {"weight": 0.50, "direction": "higher"},
            },
            "risk": {
                "volatility_3m": {"weight": 0.40, "direction": "lower"},
                "downside_volatility_3m": {"weight": 0.25, "direction": "lower"},
                "max_drawdown_1y": {"weight": 0.35, "direction": "higher"},
            },
        },
        "scoring": {
            "missing_value_score": 50.0,
            "shrink_for_missing_data": True,
            "minimum_coverage": 0.45,
            "winsor_lower": 0.02,
            "winsor_upper": 0.98,
        },
    },
    "backtest": {
        "rebalance_frequency": "M",
        "top_n": 10,
        "transaction_cost_bps": 10.0,
        "minimum_history_days": 260,
        "fundamental_lag_days": 1,
    },
    "paper_trading": {
        "starting_cash": 100000.0,
        "fee_bps": 10.0,
        "allow_fractional_shares": False,
    },
}


FACTOR_LABELS = {
    "roe": "Return on equity",
    "operating_margin": "Operating margin",
    "revenue_growth": "Revenue growth",
    "debt_to_equity": "Debt / equity",
    "pe": "Price / earnings",
    "pb": "Price / book",
    "ev_ebitda": "EV / EBITDA",
    "dividend_yield": "Dividend yield",
    "return_6m": "6-month return",
    "return_12_1m": "12-to-1-month return",
    "volatility_3m": "3-month volatility",
    "downside_volatility_3m": "Downside volatility",
    "max_drawdown_1y": "1-year maximum drawdown",
}


PERCENT_FACTORS = {
    "roe",
    "operating_margin",
    "revenue_growth",
    "dividend_yield",
    "return_6m",
    "return_12_1m",
    "volatility_3m",
    "downside_volatility_3m",
    "max_drawdown_1y",
}


def default_config() -> Dict[str, Any]:
    """Return a safe, mutable copy of the default methodology."""

    return deepcopy(DEFAULT_CONFIG)


def _normalise_weights(items: Mapping[str, float], label: str) -> Dict[str, float]:
    weights = {key: float(value) for key, value in items.items()}
    if any(value < 0 for value in weights.values()):
        raise ValueError("{} weights cannot be negative".format(label))
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("{} weights must have a positive total".format(label))
    return {key: value / total for key, value in weights.items()}


def validate_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and normalise a methodology configuration.

    The returned object is a deep copy; the input is never mutated. Category and
    factor weights may use any positive scale and are normalised to sum to one.
    """

    result = deepcopy(dict(config))
    methodology = result.get("methodology")
    if not isinstance(methodology, dict):
        raise ValueError("config.methodology must be an object")

    categories = methodology.get("category_weights", {})
    factors = methodology.get("factors", {})
    if not categories or not factors:
        raise ValueError("methodology needs category_weights and factors")
    missing_categories = set(categories) - set(factors)
    if missing_categories:
        raise ValueError(
            "No factor definitions for: {}".format(", ".join(sorted(missing_categories)))
        )
    methodology["category_weights"] = _normalise_weights(categories, "Category")

    for category, definitions in factors.items():
        if not isinstance(definitions, dict) or not definitions:
            raise ValueError("Category '{}' has no factors".format(category))
        factor_weights = {
            name: definition.get("weight", 0.0)
            for name, definition in definitions.items()
        }
        normalised = _normalise_weights(factor_weights, "{} factor".format(category))
        for name, definition in definitions.items():
            direction = definition.get("direction")
            if direction not in ("higher", "lower"):
                raise ValueError(
                    "Factor '{}.{}' direction must be higher or lower".format(
                        category, name
                    )
                )
            definition["weight"] = normalised[name]

    scoring = methodology.setdefault("scoring", {})
    scoring.setdefault("missing_value_score", 50.0)
    scoring.setdefault("shrink_for_missing_data", True)
    scoring.setdefault("minimum_coverage", 0.45)
    scoring.setdefault("winsor_lower", 0.02)
    scoring.setdefault("winsor_upper", 0.98)
    lower = float(scoring["winsor_lower"])
    upper = float(scoring["winsor_upper"])
    if not 0 <= lower < upper <= 1:
        raise ValueError("Winsor limits must satisfy 0 <= lower < upper <= 1")
    if not 0 <= float(scoring["minimum_coverage"]) <= 1:
        raise ValueError("minimum_coverage must be between 0 and 1")

    return result


def enabled_factor_names(config: Mapping[str, Any]) -> Iterable[str]:
    for definitions in config["methodology"]["factors"].values():
        for name, definition in definitions.items():
            if float(definition.get("weight", 0.0)) > 0:
                yield name
