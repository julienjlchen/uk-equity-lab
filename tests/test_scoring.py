from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from uk_equity_lab.config import DEFAULT_CONFIG
from uk_equity_lab.scoring import calculate_price_features, explain_row, score_universe


def test_factor_directions_and_rank_are_consistent():
    features = pd.DataFrame(
        {
            "ticker": ["GOOD.L", "MID.L", "WEAK.L"],
            "roe": [0.30, 0.15, 0.02],
            "operating_margin": [0.30, 0.15, 0.02],
            "revenue_growth": [0.20, 0.08, -0.10],
            "debt_to_equity": [0.1, 0.7, 2.0],
            "pe": [8, 15, 30],
            "pb": [1, 2, 5],
            "ev_ebitda": [5, 10, 20],
            "dividend_yield": [0.06, 0.03, 0.0],
            "return_6m": [0.25, 0.05, -0.20],
            "return_12_1m": [0.30, 0.02, -0.30],
            "volatility_3m": [0.10, 0.20, 0.40],
            "downside_volatility_3m": [0.07, 0.15, 0.35],
            "max_drawdown_1y": [-0.08, -0.20, -0.50],
        }
    )
    result = score_universe(features, DEFAULT_CONFIG).set_index("ticker")
    assert result.loc["GOOD.L", "overall_score"] > result.loc["MID.L", "overall_score"]
    assert result.loc["MID.L", "overall_score"] > result.loc["WEAK.L", "overall_score"]
    assert result.loc["GOOD.L", "rank"] == 1
    assert result.loc["GOOD.L", "pe__score"] > result.loc["WEAK.L", "pe__score"]
    assert "Ranks well" in explain_row(result.loc["GOOD.L"], DEFAULT_CONFIG)["summary"]


def test_missing_data_shrinks_score_toward_neutral():
    features = pd.DataFrame(
        {
            "ticker": ["FULL.L", "EMPTY.L"],
            "roe": [0.2, np.nan],
            "operating_margin": [0.2, np.nan],
            "revenue_growth": [0.2, np.nan],
            "debt_to_equity": [0.1, np.nan],
            "pe": [8, np.nan],
            "pb": [1, np.nan],
            "ev_ebitda": [5, np.nan],
            "dividend_yield": [0.05, np.nan],
            "return_6m": [0.2, np.nan],
            "return_12_1m": [0.2, np.nan],
            "volatility_3m": [0.1, np.nan],
            "downside_volatility_3m": [0.08, np.nan],
            "max_drawdown_1y": [-0.1, np.nan],
        }
    )
    result = score_universe(features, DEFAULT_CONFIG).set_index("ticker")
    assert result.loc["EMPTY.L", "overall_score"] == 50
    assert result.loc["EMPTY.L", "coverage"] == 0
    assert not bool(result.loc["EMPTY.L", "eligible"])


def test_price_features_use_only_data_up_to_as_of():
    dates = pd.bdate_range("2022-01-03", periods=300)
    prices = pd.DataFrame({"A.L": np.linspace(10, 20, 300)}, index=dates)
    first = calculate_price_features(prices, as_of=dates[-10]).set_index("ticker")
    mutated = prices.copy()
    mutated.loc[dates[-9]:, "A.L"] = 1000
    second = calculate_price_features(mutated, as_of=dates[-10]).set_index("ticker")
    assert first.loc["A.L", "return_6m"] == second.loc["A.L", "return_6m"]


def test_price_only_coverage_meets_default_threshold():
    features = pd.DataFrame(
        {
            "ticker": ["A.L", "B.L"],
            "return_6m": [0.2, -0.1],
            "return_12_1m": [0.1, -0.2],
            "volatility_3m": [0.1, 0.3],
            "downside_volatility_3m": [0.08, 0.25],
            "max_drawdown_1y": [-0.1, -0.4],
        }
    )
    result = score_universe(features, DEFAULT_CONFIG)
    assert result["coverage"].iloc[0] == pytest.approx(0.45)
    assert result["eligible"].all()
