"""Cross-sectional factor calculation, scoring, and explanations."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .config import FACTOR_LABELS, PERCENT_FACTORS, validate_config


PRICE_FACTOR_COLUMNS = [
    "return_6m",
    "return_12_1m",
    "volatility_3m",
    "downside_volatility_3m",
    "max_drawdown_1y",
]


def calculate_price_features(
    close: pd.DataFrame,
    volume: Optional[pd.DataFrame] = None,
    as_of: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Calculate point-in-time momentum and risk metrics from adjusted closes."""

    if close.empty:
        return pd.DataFrame(columns=["ticker", "latest_price"] + PRICE_FACTOR_COLUMNS)
    history = close.sort_index()
    if as_of is not None:
        history = history.loc[: pd.Timestamp(as_of)]
    history = history.dropna(how="all").ffill()
    if history.empty:
        return pd.DataFrame(columns=["ticker", "latest_price"] + PRICE_FACTOR_COLUMNS)

    latest = history.iloc[-1]
    result = pd.DataFrame(index=history.columns)
    result.index.name = "ticker"
    result["latest_price"] = latest
    result["price_as_of"] = history.index[-1]

    if len(history) > 126:
        result["return_6m"] = latest / history.iloc[-127] - 1.0
    else:
        result["return_6m"] = np.nan
    if len(history) > 252:
        result["return_12_1m"] = history.iloc[-22] / history.iloc[-253] - 1.0
    else:
        result["return_12_1m"] = np.nan

    daily_returns = history.pct_change(fill_method=None)
    last_63 = daily_returns.tail(63)
    result["volatility_3m"] = last_63.std() * np.sqrt(252)
    result["downside_volatility_3m"] = (
        last_63.where(last_63 < 0).std() * np.sqrt(252)
    )
    trailing_year = history.tail(252)
    drawdowns = trailing_year / trailing_year.cummax() - 1.0
    result["max_drawdown_1y"] = drawdowns.min()

    if volume is not None and not volume.empty:
        aligned_volume = volume.reindex(index=history.index, columns=history.columns)
        result["average_value_traded_20d"] = (
            (aligned_volume * history).tail(20).mean()
        )
    return result.reset_index()


def combine_current_features(
    fundamentals: pd.DataFrame, price_features: pd.DataFrame
) -> pd.DataFrame:
    """Merge provider fundamentals with calculated price metrics."""

    if fundamentals.empty:
        return price_features.copy()
    if price_features.empty:
        return fundamentals.copy()
    return fundamentals.merge(price_features, on="ticker", how="outer")


def _percentile_score(values: pd.Series, direction: str) -> pd.Series:
    valid = values.dropna()
    output = pd.Series(np.nan, index=values.index, dtype=float)
    if valid.empty:
        return output
    if len(valid) == 1 or valid.nunique(dropna=True) == 1:
        output.loc[valid.index] = 50.0
        return output
    ascending_score = (valid.rank(method="average") - 1.0) / (len(valid) - 1.0) * 100.0
    output.loc[valid.index] = ascending_score if direction == "higher" else 100.0 - ascending_score
    return output


def score_universe(features: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Score a universe from 0-100 using transparent cross-sectional percentiles."""

    config = validate_config(config)
    if "ticker" not in features:
        raise ValueError("features must contain a ticker column")
    scored = features.copy().drop_duplicates("ticker", keep="last")
    scored["ticker"] = scored["ticker"].astype(str)
    methodology = config["methodology"]
    category_weights = methodology["category_weights"]
    definitions = methodology["factors"]
    scoring_config = methodology["scoring"]
    missing_score = float(scoring_config["missing_value_score"])
    lower = float(scoring_config["winsor_lower"])
    upper = float(scoring_config["winsor_upper"])

    total_coverage = pd.Series(0.0, index=scored.index)
    raw_overall = pd.Series(0.0, index=scored.index)

    for category, category_weight in category_weights.items():
        category_score = pd.Series(0.0, index=scored.index)
        category_coverage = pd.Series(0.0, index=scored.index)
        for factor, definition in definitions[category].items():
            weight = float(definition["weight"])
            if factor not in scored:
                raw = pd.Series(np.nan, index=scored.index, dtype=float)
                scored[factor] = raw
            else:
                raw = pd.to_numeric(scored[factor], errors="coerce")
            if definition.get("positive_only"):
                raw = raw.where(raw > 0)
                scored[factor] = raw
            valid = raw.dropna()
            winsorised = raw.copy()
            if len(valid) >= 5:
                lo, hi = valid.quantile([lower, upper])
                winsorised = raw.clip(lo, hi)
            factor_score = _percentile_score(winsorised, definition["direction"])
            score_column = "{}__score".format(factor)
            scored[score_column] = factor_score
            category_score += weight * factor_score.fillna(missing_score)
            category_coverage += weight * raw.notna().astype(float)

        scored["{}_score".format(category)] = category_score
        scored["{}_coverage".format(category)] = category_coverage
        raw_overall += float(category_weight) * category_score
        total_coverage += float(category_weight) * category_coverage

    scored["raw_score"] = raw_overall
    scored["coverage"] = total_coverage
    if scoring_config["shrink_for_missing_data"]:
        scored["overall_score"] = missing_score + (
            raw_overall - missing_score
        ) * total_coverage
    else:
        scored["overall_score"] = raw_overall
    # The tolerance prevents a conceptual 0.30 + 0.15 coverage from failing a
    # 0.45 threshold only because of binary floating-point representation.
    scored["eligible"] = scored["coverage"] + 1e-12 >= float(
        scoring_config["minimum_coverage"]
    )
    scored["rank"] = np.nan
    eligible = scored["eligible"] & scored["overall_score"].notna()
    scored.loc[eligible, "rank"] = (
        scored.loc[eligible, "overall_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return scored.sort_values(
        ["eligible", "overall_score", "ticker"], ascending=[False, False, True]
    ).reset_index(drop=True)


def _format_raw(factor: str, value: Any) -> str:
    if pd.isna(value):
        return "missing"
    number = float(value)
    if factor in PERCENT_FACTORS:
        return "{:.1f}%".format(number * 100)
    if factor == "debt_to_equity":
        return "{:.2f}x".format(number)
    return "{:.2f}x".format(number)


def explain_row(row: pd.Series, config: Mapping[str, Any]) -> Dict[str, Any]:
    """Explain a score as strengths, weaknesses, category scores, and coverage."""

    config = validate_config(config)
    contributions: List[Tuple[float, str]] = []
    methodology = config["methodology"]
    for category, category_weight in methodology["category_weights"].items():
        for factor, definition in methodology["factors"][category].items():
            score = row.get("{}__score".format(factor), np.nan)
            raw = row.get(factor, np.nan)
            if pd.isna(score):
                continue
            contribution = (
                float(category_weight)
                * float(definition["weight"])
                * (float(score) - 50.0)
            )
            wording = "{} {} (peer score {:.0f})".format(
                FACTOR_LABELS.get(factor, factor), _format_raw(factor, raw), score
            )
            contributions.append((contribution, wording))
    contributions.sort(key=lambda item: item[0], reverse=True)
    strengths = [text for value, text in contributions if value > 0][:3]
    weaknesses = [text for value, text in reversed(contributions) if value < 0][:3]
    categories = {
        category: float(row.get("{}_score".format(category), np.nan))
        for category in methodology["category_weights"]
    }
    coverage = float(row.get("coverage", 0.0))
    if strengths:
        summary = "Ranks well mainly because {}.".format(", and ".join(strengths[:2]))
    else:
        summary = "No measured factor is above the peer median."
    if weaknesses:
        summary += " The main offsets are {}.".format(", and ".join(weaknesses[:2]))
    if coverage < 0.8:
        summary += " Confidence is reduced because factor coverage is {:.0f}%.".format(
            coverage * 100
        )
    return {
        "summary": summary,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "categories": categories,
        "coverage": coverage,
    }


def build_explanation_column(scored: pd.DataFrame, config: Mapping[str, Any]) -> pd.Series:
    return scored.apply(lambda row: explain_row(row, config)["summary"], axis=1)
