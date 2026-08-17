"""A local, broker-free paper-trading ledger."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


TRADE_COLUMNS = ["id", "timestamp", "ticker", "side", "quantity", "price", "fee", "note"]


@dataclass
class PortfolioState:
    cash: float
    positions: pd.DataFrame
    total_value: float
    total_pnl: float
    realised_pnl: float


class PaperLedger:
    """Append-only local ledger. It has deliberately no broker integration."""

    def __init__(self, path: Path, starting_cash: float, fee_bps: float = 0.0):
        self.path = Path(path)
        self.starting_cash = float(starting_cash)
        self.fee_bps = float(fee_bps)

    def trades(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=TRADE_COLUMNS)
        frame = pd.read_csv(self.path)
        for column in TRADE_COLUMNS:
            if column not in frame:
                frame[column] = np.nan
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        return frame[TRADE_COLUMNS].sort_values("timestamp").reset_index(drop=True)

    def state(self, prices: Optional[Dict[str, float]] = None) -> PortfolioState:
        prices = prices or {}
        quantities: Dict[str, float] = {}
        costs: Dict[str, float] = {}
        realised = 0.0
        cash = self.starting_cash
        for _, trade in self.trades().iterrows():
            ticker = str(trade["ticker"])
            quantity = float(trade["quantity"])
            price = float(trade["price"])
            fee = float(trade["fee"])
            old_quantity = quantities.get(ticker, 0.0)
            old_cost = costs.get(ticker, 0.0)
            if trade["side"] == "BUY":
                cash -= quantity * price + fee
                new_quantity = old_quantity + quantity
                costs[ticker] = (
                    (old_quantity * old_cost + quantity * price + fee) / new_quantity
                    if new_quantity
                    else 0.0
                )
                quantities[ticker] = new_quantity
            else:
                cash += quantity * price - fee
                realised += quantity * (price - old_cost) - fee
                quantities[ticker] = old_quantity - quantity
                if quantities[ticker] <= 1e-12:
                    quantities[ticker] = 0.0
                    costs[ticker] = 0.0

        rows = []
        for ticker, quantity in quantities.items():
            if quantity <= 1e-12:
                continue
            average_cost = costs[ticker]
            market_price = float(prices.get(ticker, average_cost))
            market_value = quantity * market_price
            unrealised = quantity * (market_price - average_cost)
            rows.append(
                {
                    "ticker": ticker,
                    "quantity": quantity,
                    "average_cost": average_cost,
                    "latest_price": market_price,
                    "market_value": market_value,
                    "unrealised_pnl": unrealised,
                }
            )
        positions = pd.DataFrame(rows)
        if not positions.empty:
            total_value = cash + float(positions["market_value"].sum())
            positions["weight"] = positions["market_value"] / total_value
            positions = positions.sort_values("market_value", ascending=False)
        else:
            total_value = cash
            positions = pd.DataFrame(
                columns=[
                    "ticker",
                    "quantity",
                    "average_cost",
                    "latest_price",
                    "market_value",
                    "unrealised_pnl",
                    "weight",
                ]
            )
        return PortfolioState(
            cash=cash,
            positions=positions,
            total_value=total_value,
            total_pnl=total_value - self.starting_cash,
            realised_pnl=realised,
        )

    def record_trade(
        self,
        ticker: str,
        side: str,
        quantity: float,
        price: float,
        note: str = "",
        timestamp: Optional[pd.Timestamp] = None,
    ) -> pd.Series:
        ticker = ticker.strip().upper()
        side = side.strip().upper()
        quantity = float(quantity)
        price = float(price)
        if side not in {"BUY", "SELL"}:
            raise ValueError("Side must be BUY or SELL")
        if not ticker or quantity <= 0 or price <= 0:
            raise ValueError("Ticker, quantity and price must be positive")
        current = self.state()
        fee = quantity * price * self.fee_bps / 10000.0
        if side == "BUY" and quantity * price + fee > current.cash + 1e-8:
            raise ValueError("Paper order exceeds available cash")
        if side == "SELL":
            owned = 0.0
            if not current.positions.empty:
                match = current.positions.loc[current.positions["ticker"] == ticker, "quantity"]
                owned = float(match.iloc[0]) if not match.empty else 0.0
            if quantity > owned + 1e-8:
                raise ValueError("Paper order would create a short position")

        row = pd.Series(
            {
                "id": uuid.uuid4().hex,
                "timestamp": timestamp or pd.Timestamp.now(tz="UTC"),
                "ticker": ticker,
                "side": side,
                "quantity": quantity,
                "price": price,
                "fee": fee,
                "note": note,
            }
        )
        trades = pd.concat([self.trades(), row.to_frame().T], ignore_index=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        trades.to_csv(temporary, index=False)
        os.replace(temporary, self.path)
        return row

    def reset(self) -> None:
        if self.path.exists():
            self.path.unlink()

    def restore(self, source: object) -> None:
        """Validate and atomically restore an exported trade-ledger CSV."""

        try:
            frame = pd.read_csv(source)
        except Exception as exc:
            raise ValueError("Could not read the paper-ledger CSV: {}".format(exc))
        missing = set(TRADE_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(
                "Paper-ledger CSV is missing: {}".format(", ".join(sorted(missing)))
            )
        frame = frame[TRADE_COLUMNS].copy()
        frame["side"] = frame["side"].astype(str).str.upper()
        frame["ticker"] = frame["ticker"].astype(str).str.upper()
        frame["quantity"] = pd.to_numeric(frame["quantity"], errors="coerce")
        frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
        frame["fee"] = pd.to_numeric(frame["fee"], errors="coerce")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        invalid = (
            ~frame["side"].isin(["BUY", "SELL"])
            | frame["ticker"].eq("")
            | frame["quantity"].le(0)
            | frame["price"].le(0)
            | frame["fee"].lt(0)
            | frame[["quantity", "price", "fee", "timestamp"]].isna().any(axis=1)
        )
        if invalid.any():
            raise ValueError("Paper-ledger CSV contains invalid trade rows")

        # Replay the file before replacing anything to enforce no-short and
        # no-overspend invariants on imported history.
        cash = self.starting_cash
        holdings: Dict[str, float] = {}
        for _, trade in frame.sort_values("timestamp").iterrows():
            ticker = trade["ticker"]
            quantity = float(trade["quantity"])
            notional = quantity * float(trade["price"])
            fee = float(trade["fee"])
            if trade["side"] == "BUY":
                cash -= notional + fee
                holdings[ticker] = holdings.get(ticker, 0.0) + quantity
                if cash < -1e-8:
                    raise ValueError("Imported ledger exceeds its paper cash balance")
            else:
                holdings[ticker] = holdings.get(ticker, 0.0) - quantity
                if holdings[ticker] < -1e-8:
                    raise ValueError("Imported ledger creates a short position")
                cash += notional - fee

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        frame.sort_values("timestamp").to_csv(temporary, index=False)
        os.replace(temporary, self.path)


def equal_weight_order_plan(
    tickers: list,
    prices: Dict[str, float],
    cash: float,
    allow_fractional: bool = False,
) -> pd.DataFrame:
    """Create a cash-only equal-weight BUY plan (it never executes trades)."""

    valid = [ticker for ticker in tickers if prices.get(ticker, 0) > 0]
    if not valid or cash <= 0:
        return pd.DataFrame(columns=["ticker", "price", "quantity", "notional"])
    allocation = cash / len(valid)
    rows = []
    for ticker in valid:
        price = float(prices[ticker])
        quantity = allocation / price
        if not allow_fractional:
            quantity = np.floor(quantity)
        if quantity <= 0:
            continue
        rows.append(
            {
                "ticker": ticker,
                "price": price,
                "quantity": quantity,
                "notional": quantity * price,
            }
        )
    return pd.DataFrame(rows)
