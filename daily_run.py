"""Headless daily ranking job for cron or a task scheduler."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import yaml

from uk_equity_lab.config import default_config, validate_config
from uk_equity_lab.data import fetch_live_bundle
from uk_equity_lab.reporting import build_markdown_report
from uk_equity_lab.scoring import (
    build_explanation_column,
    calculate_price_features,
    combine_current_features,
    score_universe,
)
from uk_equity_lab.universe import DEFAULT_UK_TICKERS, clean_tickers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the UK Equity Lab daily screen")
    parser.add_argument(
        "--tickers",
        default=",".join(DEFAULT_UK_TICKERS),
        help="Comma-separated London tickers; .L is added if omitted",
    )
    parser.add_argument("--config", type=Path, help="Optional methodology YAML")
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--years", type=int, default=3)
    parser.add_argument("--top", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config:
        with args.config.open("r", encoding="utf-8") as handle:
            config = validate_config(yaml.safe_load(handle))
    else:
        config = validate_config(default_config())
    tickers = clean_tickers(args.tickers)
    start = date.today() - timedelta(days=max(2, args.years) * 366)
    bundle = fetch_live_bundle(tickers, start)
    price_features = calculate_price_features(bundle.close, bundle.volume)
    features = combine_current_features(bundle.fundamentals, price_features)
    scored = score_universe(features, config)
    scored["explanation"] = build_explanation_column(scored, config)
    as_of = price_features["price_as_of"].max().date()
    report = build_markdown_report(scored, config, bundle.source_label, top_n=args.top)

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "rankings_{}.csv".format(as_of)
    report_path = args.output / "research_note_{}.md".format(as_of)
    scored.to_csv(csv_path, index=False)
    report_path.write_text(report, encoding="utf-8")
    print("Wrote {}".format(csv_path))
    print("Wrote {}".format(report_path))
    for warning in bundle.warnings:
        print("WARNING: {}".format(warning))


if __name__ == "__main__":
    main()
