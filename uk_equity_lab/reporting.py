"""Portable research-report exports."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Optional

import pandas as pd

from .scoring import explain_row


def build_markdown_report(
    scored: pd.DataFrame,
    config: Mapping[str, Any],
    source_label: str,
    top_n: int = 10,
    backtest_metrics: Optional[Mapping[str, float]] = None,
) -> str:
    """Build a plain-language, auditable daily research note."""

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    eligible = scored.loc[scored["eligible"]].head(top_n)
    lines = [
        "# UK Equity Lab — daily research shortlist",
        "",
        "Generated: {}".format(generated),
        "",
        "Data: {}".format(source_label),
        "",
        "> Research and paper trading only. This is not personalised investment advice.",
        "",
        "## Ranked shortlist",
        "",
    ]
    if eligible.empty:
        lines.append("No shares met the configured coverage threshold.")
    for _, row in eligible.iterrows():
        name = row.get("company_name")
        label = "{} ({})".format(name, row["ticker"]) if pd.notna(name) else row["ticker"]
        explanation = explain_row(row, config)
        lines.extend(
            [
                "### {}. {} — {:.1f}/100".format(int(row["rank"]), label, row["overall_score"]),
                "",
                explanation["summary"],
                "",
                "Category scores: fundamentals {:.0f}, valuation {:.0f}, momentum {:.0f}, risk {:.0f}. "
                "Data coverage: {:.0f}%.".format(
                    row.get("fundamentals_score", float("nan")),
                    row.get("valuation_score", float("nan")),
                    row.get("momentum_score", float("nan")),
                    row.get("risk_score", float("nan")),
                    row.get("coverage", 0) * 100,
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Methodology",
            "",
            "Inputs are winsorised cross-sectionally, converted to 0–100 peer percentiles, "
            "and aggregated using the configured factor and category weights. Missing factors "
            "receive a neutral score and optionally shrink the final score toward 50.",
            "",
        ]
    )
    weights = config["methodology"]["category_weights"]
    for category, weight in weights.items():
        lines.append("- {}: {:.0f}%".format(category.title(), weight * 100))
    if backtest_metrics:
        lines.extend(["", "## Walk-forward backtest", ""])
        labels = {
            "total_return": "Total return",
            "annualised_return": "Annualised return",
            "annualised_volatility": "Annualised volatility",
            "sharpe": "Sharpe ratio (0% cash-rate assumption)",
            "max_drawdown": "Maximum drawdown",
            "average_turnover": "Average rebalance turnover",
        }
        for key, label in labels.items():
            if key in backtest_metrics and pd.notna(backtest_metrics[key]):
                value = float(backtest_metrics[key])
                if key == "sharpe":
                    rendered = "{:.2f}".format(value)
                else:
                    rendered = "{:.1f}%".format(value * 100)
                lines.append("- {}: {}".format(label, rendered))
    lines.extend(
        [
            "",
            "## Important limitations",
            "",
            "The rank is a research prioritisation tool, not a forecast or recommendation. "
            "Current provider fundamentals are not valid historical observations. Full-factor "
            "backtests require point-in-time snapshots, and historical universes are required to "
            "remove survivorship bias. Taxes, bid/ask spread, market impact and data errors may "
            "not be captured.",
            "",
        ]
    )
    return "\n".join(lines)
