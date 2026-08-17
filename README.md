# UK Equity Lab

UK Equity Lab is a Streamlit research application for ranking London-listed shares using fundamentals, valuation, momentum and risk. It includes an auditable walk-forward backtest, downloadable research notes and a strictly local paper-trading ledger. It has no broker integration and cannot place real orders.

## What is included

- Current cross-sectional ranking on a 0–100 scale
- Configurable category weights, factor weights, directions, missing-data rules and coverage threshold
- Fundamentals: return on equity, operating margin, revenue growth and debt/equity
- Valuation: P/E, price/book, EV/EBITDA and dividend yield
- Momentum: six-month and 12-to-1-month returns
- Risk: three-month volatility, downside volatility and one-year maximum drawdown
- Plain-language strengths, weaknesses, category scores and data coverage for every share
- External analyst consensus as unscored context
- Walk-forward selection at a formation close and simulated execution at the next session close
- Point-in-time snapshot upload for honest historical fundamental/valuation tests
- Configurable rebalance frequency, shortlist size and transaction costs
- Local cash/position ledger with no shorting or overspending
- Offline fictional demo data, so the entire workflow can be evaluated safely
- CSV, Markdown research-note and YAML methodology exports

## Quick start

Python 3.9 or newer is supported.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app starts in **Safe demo** mode. For actual London symbols, select **Live market data**, edit the universe and choose **Load / refresh live data**. Yahoo Finance is used as a convenient research feed; its values can be delayed, missing or restated and must be verified against company filings before any real-world decision.

## Free phone-accessible hosting

The repository is ready for Streamlit Community Cloud, which can host the app at a mobile-accessible `streamlit.app` address:

1. Put this project in a GitHub repository.
2. Sign in to Streamlit Community Cloud with GitHub and choose **Create app**.
3. Select the repository, `main` branch and `app.py` entrypoint.
4. Keep the app private unless you intentionally want to share it.

The production dependencies are declared in `requirements.txt`; no secrets are required. Free instances can sleep after inactivity and local files are not durable across all restarts. The paper-ledger tab therefore supports CSV download and restore. A deployed server-side ledger is shared between viewers, so private deployment is strongly recommended.

## Daily headless run

The same scoring engine can produce dated files without starting Streamlit:

```bash
source .venv/bin/activate
python daily_run.py \
  --config config/default_methodology.yaml \
  --tickers AZN.L,BP.L,GSK.L,HSBA.L,SHEL.L,ULVR.L \
  --output output
```

This writes `rankings_YYYY-MM-DD.csv` and `research_note_YYYY-MM-DD.md`. Schedule that command with cron, launchd, GitHub Actions or another scheduler appropriate to the deployment. A daily run refreshes research files only; it does not create even simulated trades automatically.

## Methodology

For each run, valid raw observations are lightly winsorised and converted to cross-sectional peer percentiles. A higher-is-better factor maps the largest value toward 100; a lower-is-better factor reverses that rank. Factor percentiles are combined into category scores and then the overall score.

```text
category score = Σ(factor percentile × normalised factor weight)
raw score      = Σ(category score × normalised category weight)
final score    = 50 + (raw score − 50) × weighted data coverage
```

The last confidence adjustment is configurable. It prevents a sparsely covered share from looking unusually attractive merely because its inconvenient data is missing. Shares below the configured coverage floor remain visible for audit but are not ranked as eligible.

The configuration file is [config/default_methodology.yaml](config/default_methodology.yaml). The app also has a form editor and YAML/JSON import/export.

## Backtest integrity

Price signals use only adjusted closes on or before each formation date. The portfolio trades at the next available close, then holds an equal-weight selection until the next execution. Turnover costs are deducted from paper equity. Every formation date, execution date, selected ticker, score and coverage value is exportable.

Current fundamentals must never be pasted backward through history. In live mode, if no dated history is uploaded, the backtest uses momentum and risk only; unavailable categories score neutral and reduce coverage. To test the whole model, upload point-in-time observations with this shape:

```csv
as_of,ticker,roe,operating_margin,revenue_growth,debt_to_equity,pe,pb,ev_ebitda,dividend_yield
2024-03-18,EXAMPLE.L,0.18,0.14,0.07,0.62,13.4,1.8,8.1,0.035
```

`as_of` means the date the information became publicly usable, not the fiscal period end. The engine selects the latest snapshot before each formation cutoff. Demo mode includes simulated quarterly point-in-time snapshots.

For research-grade evidence you should also provide a historical constituent universe. Testing today's members through the past creates survivorship bias. The supplied starter list is convenient, not a historical FTSE constituent database.

## Paper ledger

The ledger is stored under `.app_data/`, which is git-ignored. It records only local simulated buys and sells. Buys above available paper cash and sells above owned quantity are rejected. The model allocation preview does nothing until the explicit simulation checkbox and button are used.

Paper trading still omits important realities: varying spreads, slippage, taxes, capacity, partial fills, market impact and data corrections. It is an educational record, not evidence that an order could have filled.

## Tests

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

The suite covers configuration validation, factor direction, missing-data treatment, point-in-time price calculation, next-session execution, cost deduction, ledger cash/short guardrails and ticker parsing.

## Project layout

```text
app.py                         Streamlit interface
daily_run.py                   Schedulable headless ranking job
config/default_methodology.yaml
uk_equity_lab/config.py        Configuration validation
uk_equity_lab/data.py          Live provider and fictional demo generator
uk_equity_lab/scoring.py       Features, peer scoring and explanations
uk_equity_lab/backtest.py      Walk-forward portfolio simulation
uk_equity_lab/paper.py         Local broker-free paper ledger
uk_equity_lab/reporting.py     Markdown research-note export
tests/                         Unit tests
```

## Disclaimer

This software is a transparent research and educational tool. It is not investment advice, a recommendation, or an offer to buy or sell a security. Scores are relative, model-dependent and sensitive to the universe and data. Past simulated performance does not predict future results.
