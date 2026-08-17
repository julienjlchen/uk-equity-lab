"""Market/fundamental data access and deterministic demo data."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class DataBundle:
    close: pd.DataFrame
    volume: pd.DataFrame
    fundamentals: pd.DataFrame
    factor_snapshots: pd.DataFrame
    source_label: str
    warnings: List[str]


FUNDAMENTAL_COLUMNS = [
    "ticker",
    "company_name",
    "sector",
    "roe",
    "operating_margin",
    "revenue_growth",
    "debt_to_equity",
    "pe",
    "pb",
    "ev_ebitda",
    "dividend_yield",
    "market_cap",
    "analyst_rating",
    "analyst_count",
    "recommendation",
    "currency",
]


def _safe_number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _normalise_debt_to_equity(value: Any) -> float:
    """Yahoo commonly returns debt/equity as a percentage (e.g. 75 for 0.75)."""

    number = _safe_number(value)
    if np.isnan(number):
        return number
    return number / 100.0 if abs(number) > 10 else number


def _normalise_dividend_yield(value: Any) -> float:
    """Accept provider versions that return either 0.0479 or 4.79 for 4.79%."""

    number = _safe_number(value)
    if np.isnan(number):
        return number
    return number / 100.0 if abs(number) > 1 else number


def _extract_info(ticker: str) -> Dict[str, Any]:
    import yfinance as yf

    info = yf.Ticker(ticker).get_info()
    return {
        "ticker": ticker,
        "company_name": info.get("shortName") or info.get("longName") or ticker,
        "sector": info.get("sector") or "Unclassified",
        "roe": _safe_number(info.get("returnOnEquity")),
        "operating_margin": _safe_number(info.get("operatingMargins")),
        "revenue_growth": _safe_number(info.get("revenueGrowth")),
        "debt_to_equity": _normalise_debt_to_equity(info.get("debtToEquity")),
        "pe": _safe_number(info.get("trailingPE")),
        "pb": _safe_number(info.get("priceToBook")),
        "ev_ebitda": _safe_number(info.get("enterpriseToEbitda")),
        "dividend_yield": _normalise_dividend_yield(info.get("dividendYield")),
        "market_cap": _safe_number(info.get("marketCap")),
        "analyst_rating": _safe_number(info.get("recommendationMean")),
        "analyst_count": _safe_number(info.get("numberOfAnalystOpinions")),
        "recommendation": info.get("recommendationKey") or "not covered",
        "currency": info.get("currency") or "GBp",
    }


def fetch_fundamentals(tickers: Sequence[str], workers: int = 8) -> Tuple[pd.DataFrame, List[str]]:
    """Fetch current company metrics concurrently from Yahoo Finance."""

    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    worker_count = max(1, min(workers, len(tickers)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(_extract_info, ticker): ticker for ticker in tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:  # providers fail per symbol; keep the screen usable
                warnings.append("{} fundamentals unavailable: {}".format(ticker, exc))
                rows.append(
                    {
                        "ticker": ticker,
                        "company_name": ticker,
                        "sector": "Unclassified",
                        "recommendation": "not covered",
                        "currency": "GBp",
                    }
                )
    frame = pd.DataFrame(rows)
    for column in FUNDAMENTAL_COLUMNS:
        if column not in frame:
            frame[column] = np.nan
    return frame[FUNDAMENTAL_COLUMNS].sort_values("ticker").reset_index(drop=True), warnings


def _extract_download_field(raw: pd.DataFrame, field: str, tickers: Sequence[str]) -> pd.DataFrame:
    if isinstance(raw.columns, pd.MultiIndex):
        if field in raw.columns.get_level_values(0):
            result = raw[field].copy()
        elif field in raw.columns.get_level_values(1):
            result = raw.xs(field, axis=1, level=1).copy()
        else:
            return pd.DataFrame(index=raw.index)
    else:
        if field not in raw:
            return pd.DataFrame(index=raw.index)
        result = raw[[field]].copy()
        result.columns = [tickers[0]]
    if isinstance(result, pd.Series):
        result = result.to_frame(name=tickers[0])
    result.columns = [str(column) for column in result.columns]
    return result.reindex(columns=list(tickers))


def fetch_prices(
    tickers: Sequence[str], start: date, end: Optional[date] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download split/dividend-adjusted closes and unadjusted volume."""

    import yfinance as yf

    if not tickers:
        raise ValueError("At least one ticker is required")
    end = end or date.today()
    raw = yf.download(
        list(tickers),
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=True,
        group_by="column",
    )
    if raw.empty:
        raise RuntimeError("The market-data provider returned no prices")
    close = _extract_download_field(raw, "Close", tickers).dropna(how="all")
    volume = _extract_download_field(raw, "Volume", tickers).reindex(close.index)
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    volume.index = pd.DatetimeIndex(volume.index).tz_localize(None)
    return close.sort_index(), volume.sort_index()


def fetch_live_bundle(
    tickers: Sequence[str], start: date, end: Optional[date] = None
) -> DataBundle:
    close, volume = fetch_prices(tickers, start, end)
    fundamentals, warnings = fetch_fundamentals(tickers)

    # London quotes from Yahoo may be denominated in pence (GBp/GBX). Convert
    # the complete series to GBP so paper-ledger values are not 100x too large.
    currency = fundamentals.set_index("ticker")["currency"].to_dict()
    for ticker in close.columns:
        quote_currency = str(currency.get(ticker, ""))
        if quote_currency in {"GBp", "GBX", "GBpence"}:
            close[ticker] = close[ticker] / 100.0

    unavailable = [ticker for ticker in tickers if ticker not in close or close[ticker].dropna().empty]
    if unavailable:
        warnings.append("No usable price history for: {}".format(", ".join(unavailable)))
    return DataBundle(
        close=close.dropna(axis=1, how="all"),
        volume=volume.reindex(columns=close.columns),
        fundamentals=fundamentals,
        factor_snapshots=pd.DataFrame(),
        source_label="Yahoo Finance (current fundamentals; adjusted prices)",
        warnings=warnings,
    )


def _demo_companies() -> pd.DataFrame:
    rows = [
        ("ALB.L", "Albion Systems", "Technology"),
        ("BRK.L", "Brookfield Retail", "Consumer Cyclical"),
        ("CRN.L", "Crown Health", "Healthcare"),
        ("DOV.L", "Dover Industrials", "Industrials"),
        ("ELM.L", "Elm Utilities", "Utilities"),
        ("FRS.L", "Farsight Media", "Communication Services"),
        ("GRN.L", "Greenline Energy", "Energy"),
        ("HBR.L", "Harbour Foods", "Consumer Defensive"),
        ("IVY.L", "Ivy Financial", "Financial Services"),
        ("JUB.L", "Jubilee Mining", "Basic Materials"),
        ("KNG.L", "Kingsway Logistics", "Industrials"),
        ("LCH.L", "Larch Property", "Real Estate"),
        ("MNR.L", "Manor Telecom", "Communication Services"),
        ("NTH.L", "Northstar Software", "Technology"),
        ("ORB.L", "Orbit Aerospace", "Industrials"),
        ("PNN.L", "Pennine Water", "Utilities"),
        ("QST.L", "Quest Diagnostics UK", "Healthcare"),
        ("RWN.L", "Rowan Brands", "Consumer Defensive"),
    ]
    return pd.DataFrame(rows, columns=["ticker", "company_name", "sector"])


def generate_demo_bundle(seed: int = 1729) -> DataBundle:
    """Create deterministic synthetic UK-style data for safe offline exploration."""

    rng = np.random.default_rng(seed)
    companies = _demo_companies()
    tickers = companies["ticker"].tolist()
    dates = pd.bdate_range("2020-01-02", pd.Timestamp.today().normalize())
    n_dates, n_assets = len(dates), len(tickers)

    latent_quality = rng.normal(0, 1, n_assets)
    latent_value = rng.normal(0, 1, n_assets)
    latent_risk = np.clip(rng.normal(0.18, 0.045, n_assets), 0.10, 0.32)
    market = rng.normal(0.00018, 0.009, n_dates)
    idiosyncratic = rng.normal(0, 1, (n_dates, n_assets))
    base_drift = 0.00012 + 0.00005 * latent_quality + 0.000035 * latent_value
    daily_vol = latent_risk / np.sqrt(252)
    returns = base_drift + 0.55 * market[:, None] + idiosyncratic * daily_vol
    prices = np.exp(np.log(rng.uniform(2.5, 75, n_assets)) + np.cumsum(returns, axis=0))
    close = pd.DataFrame(prices, index=dates, columns=tickers)
    volume = pd.DataFrame(
        rng.lognormal(mean=13.2, sigma=0.7, size=(n_dates, n_assets)),
        index=dates,
        columns=tickers,
    )

    quarter_dates = pd.date_range(dates.min(), dates.max(), freq="QE")
    snapshots: List[Dict[str, Any]] = []
    for quarter_index, as_of in enumerate(quarter_dates):
        cycle = np.sin(quarter_index / 5.0)
        for idx, row in companies.iterrows():
            q_noise = rng.normal(0, 0.018)
            roe = 0.13 + 0.055 * latent_quality[idx] + q_noise
            margin = 0.15 + 0.045 * latent_quality[idx] + rng.normal(0, 0.015)
            growth = 0.055 + 0.025 * latent_quality[idx] + 0.02 * cycle + rng.normal(0, 0.025)
            leverage = max(0.05, 0.85 - 0.16 * latent_quality[idx] + rng.normal(0, 0.10))
            pe = max(4.0, 15.0 - 2.8 * latent_value[idx] + 2.0 * latent_quality[idx] + rng.normal(0, 1.0))
            pb = max(0.4, 2.2 - 0.35 * latent_value[idx] + 0.3 * latent_quality[idx] + rng.normal(0, 0.15))
            ev_ebitda = max(2.0, 9.5 - 1.4 * latent_value[idx] + 0.8 * latent_quality[idx] + rng.normal(0, 0.6))
            dividend_yield = max(0.0, 0.035 + 0.008 * latent_value[idx] - 0.004 * latent_quality[idx] + rng.normal(0, 0.004))
            snapshots.append(
                {
                    "as_of": as_of,
                    "ticker": row["ticker"],
                    "roe": roe,
                    "operating_margin": margin,
                    "revenue_growth": growth,
                    "debt_to_equity": leverage,
                    "pe": pe,
                    "pb": pb,
                    "ev_ebitda": ev_ebitda,
                    "dividend_yield": dividend_yield,
                }
            )
    factor_snapshots = pd.DataFrame(snapshots)
    latest = (
        factor_snapshots.sort_values("as_of").groupby("ticker", as_index=False).tail(1)
    )
    fundamentals = companies.merge(latest.drop(columns="as_of"), on="ticker", how="left")
    fundamentals["market_cap"] = rng.lognormal(22.4, 1.0, n_assets)
    fundamentals["analyst_rating"] = np.clip(
        3.0 - 0.35 * latent_quality - 0.25 * latent_value + rng.normal(0, 0.2, n_assets),
        1.0,
        5.0,
    )
    fundamentals["analyst_count"] = rng.integers(3, 24, n_assets)
    fundamentals["recommendation"] = pd.cut(
        fundamentals["analyst_rating"],
        bins=[0, 1.6, 2.4, 3.4, 4.2, 5.1],
        labels=["strong buy", "buy", "hold", "underperform", "sell"],
    ).astype(str)
    fundamentals["currency"] = "GBP"
    fundamentals = fundamentals.reindex(columns=FUNDAMENTAL_COLUMNS)
    return DataBundle(
        close=close,
        volume=volume,
        fundamentals=fundamentals,
        factor_snapshots=factor_snapshots,
        source_label="Synthetic demonstration data (not real securities or prices)",
        warnings=[
            "Demo mode uses fictional companies and simulated data. No output is investable."
        ],
    )


def parse_snapshot_csv(uploaded: Any) -> pd.DataFrame:
    """Validate a point-in-time factor snapshot CSV."""

    frame = pd.read_csv(uploaded)
    required = {"as_of", "ticker"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError("Snapshot CSV is missing: {}".format(", ".join(sorted(missing))))
    frame["as_of"] = pd.to_datetime(frame["as_of"], errors="coerce")
    frame["ticker"] = frame["ticker"].astype(str).str.strip().str.upper()
    frame = frame.dropna(subset=["as_of", "ticker"])
    if frame.empty:
        raise ValueError("Snapshot CSV has no valid as_of/ticker rows")
    if frame.duplicated(["as_of", "ticker"]).any():
        raise ValueError("Snapshot CSV has duplicate as_of/ticker rows")
    return frame.sort_values(["as_of", "ticker"]).reset_index(drop=True)
