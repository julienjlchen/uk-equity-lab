"""Look-ahead-aware walk-forward rank backtesting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from .scoring import calculate_price_features, score_universe


@dataclass
class BacktestResult:
    equity: pd.Series
    benchmark_equity: pd.Series
    daily_returns: pd.Series
    holdings: pd.DataFrame
    rebalances: pd.DataFrame
    metrics: Dict[str, float]
    notes: List[str]


def _formation_dates(index: pd.DatetimeIndex, frequency: str, minimum_history: int) -> List[pd.Timestamp]:
    if len(index) <= minimum_history:
        return []
    eligible = pd.Series(index[minimum_history:], index=index[minimum_history:])
    frequency = frequency.upper()
    if frequency in {"M", "ME", "MONTHLY"}:
        periods = eligible.index.to_period("M")
    elif frequency in {"Q", "QE", "QUARTERLY"}:
        periods = eligible.index.to_period("Q")
    elif frequency in {"W", "WEEKLY"}:
        periods = eligible.index.to_period("W")
    else:
        raise ValueError("Rebalance frequency must be weekly, monthly, or quarterly")
    return eligible.groupby(periods).last().tolist()


def _latest_snapshots(
    snapshots: pd.DataFrame, as_of: pd.Timestamp, lag_days: int
) -> pd.DataFrame:
    if snapshots is None or snapshots.empty:
        return pd.DataFrame(columns=["ticker"])
    cutoff = pd.Timestamp(as_of) - pd.Timedelta(days=int(lag_days))
    available = snapshots.loc[pd.to_datetime(snapshots["as_of"]) <= cutoff].copy()
    if available.empty:
        return pd.DataFrame(columns=["ticker"])
    return (
        available.sort_values("as_of")
        .groupby("ticker", as_index=False)
        .tail(1)
        .drop(columns=["as_of"], errors="ignore")
    )


def _calculate_metrics(returns: pd.Series, equity: pd.Series) -> Dict[str, float]:
    returns = returns.dropna()
    if returns.empty or equity.empty:
        return {
            "total_return": np.nan,
            "annualised_return": np.nan,
            "annualised_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
        }
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    annualised = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)
    volatility = float(returns.std() * np.sqrt(252))
    annual_mean = float(returns.mean() * 252)
    sharpe = annual_mean / volatility if volatility > 0 else np.nan
    drawdown = equity / equity.cummax() - 1.0
    return {
        "total_return": total,
        "annualised_return": annualised,
        "annualised_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
    }


def run_backtest(
    close: pd.DataFrame,
    config: Mapping[str, Any],
    snapshots: Optional[pd.DataFrame] = None,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    top_n: Optional[int] = None,
    frequency: Optional[str] = None,
    transaction_cost_bps: Optional[float] = None,
    benchmark_close: Optional[pd.Series] = None,
) -> BacktestResult:
    """Walk forward through ranks and rebalance an equal-weight paper portfolio.

    Scores are calculated at each formation close. Trades take place at the next
    available session's close, so that same-close information is never traded
    before it exists. Fundamental rows must be point-in-time snapshots whose
    ``as_of`` date represents when the information became usable.
    """

    settings = config["backtest"]
    top_n = int(top_n if top_n is not None else settings["top_n"])
    frequency = frequency or settings["rebalance_frequency"]
    cost_bps = float(
        transaction_cost_bps
        if transaction_cost_bps is not None
        else settings["transaction_cost_bps"]
    )
    minimum_history = int(settings.get("minimum_history_days", 260))
    lag_days = int(settings.get("fundamental_lag_days", 1))
    prices = close.sort_index().copy()
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None)
    prices = prices.ffill().dropna(how="all")
    if start is not None:
        calculation_start = pd.Timestamp(start) - pd.Timedelta(days=550)
        prices = prices.loc[calculation_start:]
    if end is not None:
        prices = prices.loc[: pd.Timestamp(end)]
    prices = prices.dropna(axis=1, how="all")
    if len(prices) <= minimum_history + 2:
        raise ValueError(
            "Not enough price history: need at least {} trading days".format(
                minimum_history + 3
            )
        )

    formation_dates = _formation_dates(prices.index, frequency, minimum_history)
    if start is not None:
        formation_dates = [day for day in formation_dates if day >= pd.Timestamp(start)]
    execution_map: Dict[pd.Timestamp, Dict[str, Any]] = {}
    holding_rows: List[Dict[str, Any]] = []
    notes: List[str] = []
    for formation in formation_dates:
        next_locations = prices.index[prices.index > formation]
        if next_locations.empty:
            continue
        execution = next_locations[0]
        price_features = calculate_price_features(prices, as_of=formation)
        point_in_time = _latest_snapshots(snapshots, formation, lag_days)
        # Price-derived signals are always recalculated point in time. Ignore
        # uploaded versions rather than creating ambiguous _x/_y columns.
        point_in_time = point_in_time.drop(
            columns=[
                "return_6m",
                "return_12_1m",
                "volatility_3m",
                "downside_volatility_3m",
                "max_drawdown_1y",
                "latest_price",
                "price_as_of",
            ],
            errors="ignore",
        )
        features = price_features.merge(point_in_time, on="ticker", how="left")
        scored = score_universe(features, config)
        candidates = scored.loc[scored["eligible"] & scored["latest_price"].notna()].head(top_n)
        if candidates.empty:
            continue
        selected = candidates["ticker"].tolist()
        execution_map[execution] = {
            "formation_date": formation,
            "tickers": selected,
            "scores": dict(zip(candidates["ticker"], candidates["overall_score"])),
            "coverage": float(candidates["coverage"].mean()),
        }
        for _, candidate in candidates.iterrows():
            holding_rows.append(
                {
                    "formation_date": formation,
                    "execution_date": execution,
                    "ticker": candidate["ticker"],
                    "rank": candidate["rank"],
                    "score": candidate["overall_score"],
                    "coverage": candidate["coverage"],
                }
            )

    if not execution_map:
        raise ValueError("No eligible rebalance dates or shares were found")
    if snapshots is None or snapshots.empty:
        notes.append(
            "No point-in-time fundamental snapshots were supplied. Historical ranks "
            "therefore use momentum and risk; missing categories are neutral and reduce confidence."
        )
    else:
        notes.append(
            "Fundamental/valuation inputs use the latest snapshot dated before each formation cutoff."
        )
    notes.append(
        "Ranks form at a session close and execute at the next session close; dividends/splits "
        "are represented by adjusted prices."
    )

    first_execution = min(execution_map)
    simulation_prices = prices.loc[first_execution:].copy()
    if end is not None:
        simulation_prices = simulation_prices.loc[: pd.Timestamp(end)]
    cash = 1.0
    shares: Dict[str, float] = {}
    equity_values: List[float] = []
    equity_dates: List[pd.Timestamp] = []
    rebalance_rows: List[Dict[str, Any]] = []

    for day, row in simulation_prices.iterrows():
        position_values = {
            ticker: quantity * float(row.get(ticker, np.nan))
            for ticker, quantity in shares.items()
            if pd.notna(row.get(ticker, np.nan))
        }
        portfolio_before = cash + sum(position_values.values())
        if day in execution_map:
            instruction = execution_map[day]
            chosen = [
                ticker
                for ticker in instruction["tickers"]
                if ticker in row.index and pd.notna(row[ticker]) and row[ticker] > 0
            ]
            if chosen:
                current_weights = {
                    ticker: value / portfolio_before
                    for ticker, value in position_values.items()
                }
                current_weights["__cash__"] = cash / portfolio_before
                target_weights = {ticker: 1.0 / len(chosen) for ticker in chosen}
                target_weights["__cash__"] = 0.0
                assets = set(current_weights) | set(target_weights)
                turnover = 0.5 * sum(
                    abs(target_weights.get(asset, 0.0) - current_weights.get(asset, 0.0))
                    for asset in assets
                )
                cost = portfolio_before * turnover * cost_bps / 10000.0
                investable = max(0.0, portfolio_before - cost)
                allocation = investable / len(chosen)
                shares = {ticker: allocation / float(row[ticker]) for ticker in chosen}
                cash = 0.0
                portfolio_before = investable
                rebalance_rows.append(
                    {
                        "formation_date": instruction["formation_date"],
                        "execution_date": day,
                        "holdings": len(chosen),
                        "turnover": turnover,
                        "cost": cost,
                        "average_score": float(
                            np.mean([instruction["scores"][ticker] for ticker in chosen])
                        ),
                        "average_coverage": instruction["coverage"],
                    }
                )
        equity_dates.append(day)
        equity_values.append(portfolio_before)

    equity = pd.Series(equity_values, index=equity_dates, name="Strategy")
    daily_returns = equity.pct_change().fillna(0.0).rename("Strategy return")
    if benchmark_close is None or benchmark_close.dropna().empty:
        benchmark_daily = simulation_prices.pct_change(fill_method=None).mean(axis=1).fillna(0.0)
        benchmark_equity = (1.0 + benchmark_daily).cumprod().rename("Universe equal-weight")
        notes.append("Benchmark is the daily equal-weight return of the available universe.")
    else:
        benchmark = benchmark_close.copy()
        benchmark.index = pd.DatetimeIndex(benchmark.index).tz_localize(None)
        benchmark = benchmark.sort_index().reindex(simulation_prices.index).ffill().dropna()
        benchmark_equity = (benchmark / benchmark.iloc[0]).rename("Benchmark")
    benchmark_equity = benchmark_equity.reindex(equity.index).ffill()

    metrics = _calculate_metrics(daily_returns, equity)
    benchmark_returns = benchmark_equity.pct_change().fillna(0.0)
    benchmark_metrics = _calculate_metrics(benchmark_returns, benchmark_equity)
    metrics.update(
        {"benchmark_{}".format(key): value for key, value in benchmark_metrics.items()}
    )
    rebalances = pd.DataFrame(rebalance_rows)
    if not rebalances.empty:
        metrics["average_turnover"] = float(rebalances["turnover"].mean())
        metrics["total_cost"] = float(rebalances["cost"].sum())
    return BacktestResult(
        equity=equity,
        benchmark_equity=benchmark_equity,
        daily_returns=daily_returns,
        holdings=pd.DataFrame(holding_rows),
        rebalances=rebalances,
        metrics=metrics,
        notes=notes,
    )
