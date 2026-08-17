"""Streamlit entry point for UK Equity Lab."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

from uk_equity_lab.backtest import run_backtest
from uk_equity_lab.config import FACTOR_LABELS, default_config, validate_config
from uk_equity_lab.data import (
    DataBundle,
    fetch_live_bundle,
    generate_demo_bundle,
    parse_snapshot_csv,
)
from uk_equity_lab.paper import PaperLedger, equal_weight_order_plan
from uk_equity_lab.reporting import build_markdown_report
from uk_equity_lab.scoring import (
    build_explanation_column,
    calculate_price_features,
    combine_current_features,
    explain_row,
    score_universe,
)
from uk_equity_lab.universe import DEFAULT_UK_TICKERS, clean_tickers


APP_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="UK Equity Lab",
    page_icon="🇬🇧",
    layout="wide",
    initial_sidebar_state="auto",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1500px;}
      [data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
      .lab-kicker {font-size:.78rem; letter-spacing:.12em; font-weight:700; color:#55706a; text-transform:uppercase;}
      .lab-note {padding:.8rem 1rem; border:1px solid #dce7e2; border-radius:.6rem; background:#f7faf8;}
      .score-pill {display:inline-block; padding:.2rem .65rem; border-radius:999px; background:#e1f3e8; color:#14532d; font-weight:700;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60 * 60 * 6, show_spinner=False)
def cached_live_bundle(tickers: tuple, start_iso: str) -> DataBundle:
    return fetch_live_bundle(tickers, date.fromisoformat(start_iso))


@st.cache_data(show_spinner=False)
def cached_demo_bundle() -> DataBundle:
    return generate_demo_bundle()


def pct(value: float) -> str:
    return "—" if pd.isna(value) else "{:.1f}%".format(float(value) * 100)


def gbp(value: float) -> str:
    return "—" if pd.isna(value) else "£{:,.0f}".format(float(value))


def model_view(score: float) -> str:
    if score >= 70:
        return "High-ranked research candidate"
    if score >= 55:
        return "Research watchlist"
    if score >= 45:
        return "Peer-neutral"
    return "Lower-ranked on this model"


if "methodology_config" not in st.session_state:
    st.session_state.methodology_config = default_config()

try:
    config = validate_config(st.session_state.methodology_config)
except ValueError as exc:
    st.error("Configuration error: {}".format(exc))
    st.session_state.methodology_config = default_config()
    config = validate_config(st.session_state.methodology_config)


with st.sidebar:
    st.markdown("### Data & run settings")
    source = st.radio(
        "Data source",
        ["Safe demo", "Live market data"],
        help="Demo is deterministic and offline. Live mode uses Yahoo Finance and may be delayed.",
    )
    history_years = st.slider("Price history (years)", 2, 10, 6)
    ticker_text = st.text_area(
        "UK tickers",
        value="\n".join(DEFAULT_UK_TICKERS[:25]),
        height=150,
        disabled=source == "Safe demo",
        help="London tickers use Yahoo's .L suffix; it is added automatically.",
    )
    tickers = clean_tickers(ticker_text)
    refresh_live = st.button(
        "Load / refresh live data",
        type="primary",
        width="stretch",
        disabled=source == "Safe demo" or not tickers,
    )
    if refresh_live:
        cached_live_bundle.clear()
        st.session_state.live_requested = True

    st.divider()
    st.markdown("### Point-in-time research data")
    snapshot_upload = st.file_uploader(
        "Historical factor snapshots (CSV)",
        type=["csv"],
        help="Optional. Required for a full fundamental/valuation backtest in live mode.",
    )

    st.divider()
    st.markdown("### Method configuration")
    config_upload = st.file_uploader("Load YAML or JSON", type=["yaml", "yml", "json"])
    if config_upload is not None and st.button("Apply uploaded configuration"):
        try:
            payload = config_upload.getvalue().decode("utf-8")
            loaded = (
                json.loads(payload)
                if config_upload.name.lower().endswith(".json")
                else yaml.safe_load(payload)
            )
            st.session_state.methodology_config = validate_config(loaded)
            st.success("Configuration applied")
            st.rerun()
        except Exception as exc:
            st.error("Could not apply configuration: {}".format(exc))
    config_yaml = yaml.safe_dump(config, sort_keys=False)
    st.download_button(
        "Download current methodology",
        data=config_yaml,
        file_name="uk_equity_methodology.yaml",
        mime="text/yaml",
        width="stretch",
    )
    st.caption(
        "Category weights: "
        + " · ".join(
            "{} {:.0f}%".format(name[:4].title(), weight * 100)
            for name, weight in config["methodology"]["category_weights"].items()
        )
    )


st.markdown('<div class="lab-kicker">Explainable quantitative research</div>', unsafe_allow_html=True)
st.title("UK Equity Lab")
st.caption(
    "Daily UK-share ranking, honest walk-forward testing and a broker-free paper ledger. "
    "Research only — not personalised investment advice."
)

if source == "Safe demo":
    bundle = cached_demo_bundle()
    data_key = "demo"
else:
    if not st.session_state.get("live_requested"):
        st.info("Choose the live universe in the sidebar, then select **Load / refresh live data**.")
        st.stop()
    start_date = date.today() - timedelta(days=int(history_years * 365.25) + 20)
    try:
        with st.spinner("Loading adjusted prices and current company metrics…"):
            bundle = cached_live_bundle(tuple(tickers), start_date.isoformat())
        data_key = "live"
    except Exception as exc:
        st.error("Live data could not be loaded: {}".format(exc))
        st.info("Switch to Safe demo to explore the complete application without network data.")
        st.stop()

for warning in bundle.warnings:
    st.warning(warning, icon="⚠️")

snapshots = bundle.factor_snapshots
if snapshot_upload is not None:
    try:
        snapshots = parse_snapshot_csv(snapshot_upload)
        st.success("Loaded {:,} point-in-time factor rows.".format(len(snapshots)))
    except ValueError as exc:
        st.error(str(exc))

price_features = calculate_price_features(bundle.close, bundle.volume)
features = combine_current_features(bundle.fundamentals, price_features)
scored = score_universe(features, config)
scored["explanation"] = build_explanation_column(scored, config)
scored["model_view"] = scored["overall_score"].map(model_view)
eligible_count = int(scored["eligible"].sum())
price_as_of = pd.to_datetime(price_features["price_as_of"].max()).date()

metric_cols = st.columns(4)
metric_cols[0].metric("Shares scored", "{:,}".format(len(scored)))
metric_cols[1].metric("Coverage-qualified", "{:,}".format(eligible_count))
metric_cols[2].metric("Latest price date", str(price_as_of))
metric_cols[3].metric("Median factor coverage", pct(scored["coverage"].median()))
st.caption("Source: {}".format(bundle.source_label))

rank_tab, backtest_tab, paper_tab, method_tab, data_tab = st.tabs(
    ["Ranked shortlist", "Walk-forward backtest", "Paper portfolio", "Methodology", "Data audit"]
)


with rank_tab:
    controls = st.columns([2, 1, 1])
    sectors = sorted(scored["sector"].dropna().astype(str).unique()) if "sector" in scored else []
    selected_sectors = controls[0].multiselect("Sectors", sectors, default=[])
    top_count = controls[1].selectbox("Rows", [10, 15, 25, 50], index=1)
    minimum_score = controls[2].number_input("Minimum score", 0.0, 100.0, 0.0, 1.0)
    shortlist = scored.loc[scored["eligible"] & (scored["overall_score"] >= minimum_score)].copy()
    if selected_sectors:
        shortlist = shortlist.loc[shortlist["sector"].isin(selected_sectors)]
    shortlist = shortlist.head(top_count)

    if shortlist.empty:
        st.info("No shares match these filters and the configured coverage threshold.")
    else:
        table_columns = [
            "rank",
            "ticker",
            "company_name",
            "sector",
            "latest_price",
            "overall_score",
            "fundamentals_score",
            "valuation_score",
            "momentum_score",
            "risk_score",
            "coverage",
            "model_view",
            "recommendation",
        ]
        visible_columns = [column for column in table_columns if column in shortlist]
        st.dataframe(
            shortlist[visible_columns],
            hide_index=True,
            width="stretch",
            column_config={
                "rank": st.column_config.NumberColumn("Rank", format="%d"),
                "latest_price": st.column_config.NumberColumn("Price (GBP)", format="£%.2f"),
                "overall_score": st.column_config.ProgressColumn("Overall", min_value=0, max_value=100, format="%.1f"),
                "fundamentals_score": st.column_config.NumberColumn("Fund.", format="%.0f"),
                "valuation_score": st.column_config.NumberColumn("Value", format="%.0f"),
                "momentum_score": st.column_config.NumberColumn("Momentum", format="%.0f"),
                "risk_score": st.column_config.NumberColumn("Risk", format="%.0f"),
                "coverage": st.column_config.ProgressColumn("Coverage", min_value=0, max_value=1, format="percent"),
                "recommendation": "Analyst context",
                "model_view": "Model view (not advice)",
            },
        )

        left, right = st.columns([1, 1.35])
        selected_ticker = left.selectbox("Inspect a score", shortlist["ticker"].tolist())
        selected = scored.loc[scored["ticker"] == selected_ticker].iloc[0]
        detail = explain_row(selected, config)
        name = selected.get("company_name", selected_ticker)
        left.markdown("### {}".format(name if pd.notna(name) else selected_ticker))
        left.markdown(
            '<span class="score-pill">#{:.0f} · {:.1f}/100</span>'.format(
                selected["rank"], selected["overall_score"]
            ),
            unsafe_allow_html=True,
        )
        left.write(detail["summary"])
        if detail["strengths"]:
            left.markdown("**Measured strengths**")
            for item in detail["strengths"]:
                left.write("+ " + item)
        if detail["weaknesses"]:
            left.markdown("**Measured weaknesses**")
            for item in detail["weaknesses"]:
                left.write("− " + item)
        analyst = selected.get("recommendation", "not covered")
        analyst_count = selected.get("analyst_count", np.nan)
        left.caption(
            "External analyst context: {}{}; this is displayed but not included in the default score.".format(
                analyst,
                " ({} analysts)".format(int(analyst_count)) if pd.notna(analyst_count) else "",
            )
        )

        category_frame = pd.DataFrame(
            {
                "Category": [name.title() for name in detail["categories"]],
                "Score": list(detail["categories"].values()),
            }
        )
        figure = px.bar(
            category_frame,
            x="Score",
            y="Category",
            orientation="h",
            range_x=[0, 100],
            color="Score",
            color_continuous_scale=[(0, "#b45309"), (0.5, "#d5ddd9"), (1, "#0f766e")],
        )
        figure.add_vline(x=50, line_dash="dot", line_color="#52615c")
        figure.update_layout(
            title="Category scores versus peers",
            coloraxis_showscale=False,
            height=330,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        right.plotly_chart(figure)

    report_metrics = None
    if st.session_state.get("backtest_key") == data_key:
        report_metrics = st.session_state.backtest_result.metrics
    report = build_markdown_report(
        scored, config, bundle.source_label, top_n=top_count, backtest_metrics=report_metrics
    )
    export_cols = [
        column
        for column in [
            "rank",
            "ticker",
            "company_name",
            "sector",
            "overall_score",
            "fundamentals_score",
            "valuation_score",
            "momentum_score",
            "risk_score",
            "coverage",
            "model_view",
            "explanation",
        ]
        if column in scored
    ]
    downloads = st.columns(2)
    downloads[0].download_button(
        "Download ranked CSV",
        scored[export_cols].to_csv(index=False),
        "uk_equity_rankings_{}.csv".format(price_as_of),
        "text/csv",
        width="stretch",
    )
    downloads[1].download_button(
        "Download transparent research note",
        report,
        "uk_equity_research_note_{}.md".format(price_as_of),
        "text/markdown",
        width="stretch",
    )


with backtest_tab:
    st.subheader("Walk-forward rank test")
    st.write(
        "The model forms a rank after each weekly, month-end or quarter-end close and executes "
        "at the next session's close. It holds the top-ranked shares at equal weight."
    )
    if source == "Live market data" and snapshots.empty:
        st.warning(
            "Live mode has no historical fundamentals. This run will honestly test only the "
            "price-derived momentum/risk portion, with other categories neutral. Upload dated "
            "snapshots for a full-factor test."
        )
    bt_min = bundle.close.index[min(260, len(bundle.close) - 2)].date()
    bt_max = bundle.close.index[-2].date()
    default_start = max(bt_min, (pd.Timestamp(bt_max) - pd.DateOffset(years=3)).date())
    with st.form("backtest_form"):
        bt_cols = st.columns(5)
        start = bt_cols[0].date_input("Start", default_start, min_value=bt_min, max_value=bt_max)
        end = bt_cols[1].date_input("End", bt_max, min_value=bt_min, max_value=bt_max)
        frequency_label = bt_cols[2].selectbox("Rebalance", ["Monthly", "Quarterly", "Weekly"])
        backtest_top_n = bt_cols[3].number_input("Top shares", 1, max(1, len(scored)), min(10, len(scored)))
        costs = bt_cols[4].number_input("Costs (bps / turnover)", 0.0, 250.0, float(config["backtest"]["transaction_cost_bps"]), 1.0)
        run_test = st.form_submit_button("Run walk-forward backtest", type="primary")
    if run_test:
        if start >= end:
            st.error("Backtest start must be before end.")
        else:
            frequency_code = {"Monthly": "M", "Quarterly": "Q", "Weekly": "W"}[frequency_label]
            try:
                with st.spinner("Walking through historical formation dates…"):
                    result = run_backtest(
                        bundle.close,
                        config,
                        snapshots=snapshots,
                        start=pd.Timestamp(start),
                        end=pd.Timestamp(end),
                        top_n=int(backtest_top_n),
                        frequency=frequency_code,
                        transaction_cost_bps=float(costs),
                    )
                st.session_state.backtest_result = result
                st.session_state.backtest_key = data_key
            except Exception as exc:
                st.error("Backtest could not run: {}".format(exc))
    if st.session_state.get("backtest_key") == data_key:
        result = st.session_state.backtest_result
        metrics = result.metrics
        bt_metrics = st.columns(5)
        bt_metrics[0].metric("Total return", pct(metrics["total_return"]))
        bt_metrics[1].metric("Annualised", pct(metrics["annualised_return"]))
        bt_metrics[2].metric("Volatility", pct(metrics["annualised_volatility"]))
        bt_metrics[3].metric("Sharpe (0% cash rate)", "{:.2f}".format(metrics["sharpe"]))
        bt_metrics[4].metric("Max drawdown", pct(metrics["max_drawdown"]))
        performance = pd.concat([result.equity, result.benchmark_equity], axis=1)
        performance = performance / performance.iloc[0] * 100.0
        perf_figure = go.Figure()
        perf_figure.add_trace(go.Scatter(x=performance.index, y=performance.iloc[:, 0], name="Rank strategy", line=dict(color="#0f766e", width=2.5)))
        perf_figure.add_trace(go.Scatter(x=performance.index, y=performance.iloc[:, 1], name=performance.columns[1], line=dict(color="#7c8c86", width=1.8)))
        perf_figure.update_layout(
            title="Growth of 100 (after configured transaction costs)",
            yaxis_title="Paper value",
            height=430,
            margin=dict(l=10, r=10, t=55, b=10),
            legend=dict(orientation="h"),
        )
        st.plotly_chart(perf_figure)
        for note in result.notes:
            st.caption("• " + note)
        with st.expander("Rebalance audit trail"):
            st.dataframe(result.rebalances, hide_index=True, width="stretch")
        with st.expander("Historical selections"):
            st.dataframe(result.holdings, hide_index=True, width="stretch")
        st.download_button(
            "Download backtest holdings",
            result.holdings.to_csv(index=False),
            "uk_equity_backtest_holdings.csv",
            "text/csv",
        )
    else:
        st.info("Run the test to see performance, turnover, costs and every historical selection.")


with paper_tab:
    st.subheader("Paper portfolio — no broker connection")
    st.write(
        "Trades are recorded locally at the displayed closing price. No order can reach a broker, "
        "short selling is disabled, and the ledger refuses purchases above its paper cash balance."
    )
    st.info(
        "Free cloud instances can restart and their local storage is not guaranteed to persist. "
        "Download the paper ledger after changes; you can restore that CSV here later. Keep the "
        "deployed app private because a server-side ledger is shared by everyone who can open it."
    )
    ledger = PaperLedger(
        APP_DIR / ".app_data" / "paper_trades_{}.csv".format(data_key),
        starting_cash=float(config["paper_trading"]["starting_cash"]),
        fee_bps=float(config["paper_trading"]["fee_bps"]),
    )
    latest_prices = dict(zip(price_features["ticker"], price_features["latest_price"]))
    state = ledger.state(latest_prices)
    paper_metrics = st.columns(4)
    paper_metrics[0].metric("Paper value", gbp(state.total_value))
    paper_metrics[1].metric("Cash", gbp(state.cash))
    paper_metrics[2].metric("Total paper P&L", gbp(state.total_pnl))
    paper_metrics[3].metric("Open positions", len(state.positions))

    if not state.positions.empty:
        st.dataframe(
            state.positions,
            hide_index=True,
            width="stretch",
            column_config={
                "quantity": st.column_config.NumberColumn("Shares", format="%.4f"),
                "average_cost": st.column_config.NumberColumn("Average cost", format="£%.2f"),
                "latest_price": st.column_config.NumberColumn("Latest close", format="£%.2f"),
                "market_value": st.column_config.NumberColumn("Paper value", format="£%.2f"),
                "unrealised_pnl": st.column_config.NumberColumn("Unrealised P&L", format="£%.2f"),
                "weight": st.column_config.ProgressColumn("Weight", min_value=0, max_value=1, format="percent"),
            },
        )

    order_col, plan_col = st.columns(2)
    with order_col:
        st.markdown("#### Record a manual paper trade")
        with st.form("manual_paper_trade"):
            order_ticker = st.selectbox("Ticker", scored["ticker"].tolist())
            side = st.radio("Side", ["BUY", "SELL"], horizontal=True)
            default_price = float(latest_prices.get(order_ticker, 0.01))
            quantity = st.number_input("Quantity", min_value=0.0001, value=1.0, step=1.0)
            order_price = st.number_input("Paper execution price (GBP)", min_value=0.0001, value=default_price, step=0.01, format="%.4f")
            note = st.text_input("Note", "Manual paper trade")
            record = st.form_submit_button("Record paper trade", type="primary")
        if record:
            try:
                ledger.record_trade(order_ticker, side, quantity, order_price, note)
                st.success("Paper trade recorded; no real order was sent.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    with plan_col:
        st.markdown("#### Top-rank paper allocation plan")
        plan_size = st.slider("Number of candidates", 1, min(20, max(1, eligible_count)), min(5, max(1, eligible_count)))
        plan_tickers = scored.loc[scored["eligible"], "ticker"].head(plan_size).tolist()
        fee_rate = float(config["paper_trading"]["fee_bps"]) / 10000.0
        available_before_fees = max(0.0, state.cash / (1.0 + fee_rate))
        plan = equal_weight_order_plan(
            plan_tickers,
            latest_prices,
            available_before_fees,
            allow_fractional=bool(config["paper_trading"]["allow_fractional_shares"]),
        )
        st.dataframe(plan, hide_index=True, width="stretch")
        confirm_plan = st.checkbox("I understand this records simulated trades only")
        if st.button("Execute plan in paper ledger", disabled=not confirm_plan or plan.empty):
            try:
                for _, order in plan.iterrows():
                    ledger.record_trade(
                        order["ticker"],
                        "BUY",
                        order["quantity"],
                        order["price"],
                        "Equal-weight model shortlist",
                    )
                st.success("Paper allocation recorded; no real orders were sent.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

    trades = ledger.trades()
    with st.expander("Paper trade ledger"):
        st.dataframe(trades, hide_index=True, width="stretch")
        if not trades.empty:
            st.download_button("Download ledger CSV", trades.to_csv(index=False), "paper_trades.csv", "text/csv")
        restore_upload = st.file_uploader(
            "Restore a previously downloaded paper ledger",
            type=["csv"],
            key="paper_ledger_restore",
        )
        restore_confirm = st.checkbox(
            "Replace the current paper ledger with this uploaded CSV",
            key="paper_restore_confirm",
        )
        if st.button(
            "Restore paper ledger",
            disabled=restore_upload is None or not restore_confirm,
        ):
            try:
                ledger.restore(restore_upload)
                st.success("Paper ledger restored.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        reset_confirm = st.checkbox("Confirm deletion of this local paper ledger", key="reset_paper_confirm")
        if st.button("Reset paper ledger", disabled=not reset_confirm):
            ledger.reset()
            st.rerun()


with method_tab:
    st.subheader("Configurable, auditable methodology")
    st.write(
        "Each raw factor is winsorised, ranked against the current peer universe, and mapped to "
        "0–100. A score of 50 is peer-neutral. 'Lower' direction means a smaller raw value scores "
        "better; the risk category therefore rewards lower volatility and shallower drawdowns."
    )
    edited = deepcopy(config)
    with st.form("methodology_editor"):
        st.markdown("#### Category weights")
        category_cols = st.columns(len(edited["methodology"]["category_weights"]))
        for idx, (category, value) in enumerate(edited["methodology"]["category_weights"].items()):
            edited["methodology"]["category_weights"][category] = category_cols[idx].number_input(
                category.title(), 0.0, 100.0, float(value * 100), 1.0, key="cat_{}".format(category)
            )
        st.markdown("#### Factor settings")
        for category, definitions in edited["methodology"]["factors"].items():
            st.markdown("**{}**".format(category.title()))
            for factor, definition in definitions.items():
                cols = st.columns([2.2, 1, 1])
                cols[0].write(FACTOR_LABELS.get(factor, factor))
                definition["weight"] = cols[1].number_input(
                    "Weight",
                    0.0,
                    100.0,
                    float(definition["weight"] * 100),
                    1.0,
                    key="factor_weight_{}".format(factor),
                    label_visibility="collapsed",
                )
                definition["direction"] = cols[2].selectbox(
                    "Direction",
                    ["higher", "lower"],
                    index=0 if definition["direction"] == "higher" else 1,
                    key="factor_direction_{}".format(factor),
                    label_visibility="collapsed",
                )
        scoring_settings = edited["methodology"]["scoring"]
        setting_cols = st.columns(3)
        scoring_settings["minimum_coverage"] = setting_cols[0].slider(
            "Minimum data coverage", 0.0, 1.0, float(scoring_settings["minimum_coverage"]), 0.05
        )
        scoring_settings["missing_value_score"] = setting_cols[1].number_input(
            "Neutral missing score", 0.0, 100.0, float(scoring_settings["missing_value_score"]), 1.0
        )
        scoring_settings["shrink_for_missing_data"] = setting_cols[2].checkbox(
            "Shrink incomplete scores", bool(scoring_settings["shrink_for_missing_data"])
        )
        apply_method = st.form_submit_button("Apply methodology", type="primary")
    if apply_method:
        try:
            # The editor accepts percentages; validation then normalises every group.
            st.session_state.methodology_config = validate_config(edited)
            st.success("Methodology updated and ranks recalculated.")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))

    st.markdown("#### Score formula")
    st.code(
        "factor score = cross-sectional percentile (0–100)\n"
        "category score = Σ(factor score × normalised factor weight)\n"
        "raw overall = Σ(category score × normalised category weight)\n"
        "confidence-adjusted = 50 + (raw overall − 50) × data coverage",
        language="text",
    )
    st.info(
        "External analyst consensus is context only in the default model. This avoids rewarding "
        "coverage volume and accidentally treating a current recommendation as historical data."
    )


with data_tab:
    st.subheader("Data audit and limitations")
    availability_rows = []
    for category, definitions in config["methodology"]["factors"].items():
        for factor in definitions:
            availability_rows.append(
                {
                    "category": category,
                    "factor": factor,
                    "description": FACTOR_LABELS.get(factor, factor),
                    "available_shares": int(pd.to_numeric(features.get(factor), errors="coerce").notna().sum()) if factor in features else 0,
                    "universe_shares": len(features),
                    "historical_snapshot_column": factor in snapshots.columns,
                }
            )
    st.dataframe(pd.DataFrame(availability_rows), hide_index=True, width="stretch")
    st.markdown("#### Point-in-time snapshot schema")
    st.write(
        "Upload one row per ticker and information-availability date. The backtest selects the "
        "latest row strictly before each configured cutoff. The `as_of` date should be the date "
        "the market could know the value, not the fiscal period end."
    )
    template_columns = ["as_of", "ticker"] + [
        factor
        for definitions in config["methodology"]["factors"].values()
        for factor in definitions
        if factor not in {"return_6m", "return_12_1m", "volatility_3m", "downside_volatility_3m", "max_drawdown_1y"}
    ]
    template = pd.DataFrame(columns=template_columns)
    st.download_button(
        "Download snapshot CSV template",
        template.to_csv(index=False),
        "point_in_time_factor_template.csv",
        "text/csv",
    )
    with st.expander("Raw current feature data"):
        st.dataframe(features, hide_index=True, width="stretch")
    st.markdown(
        """
        **Known limitations**

        - A current constituent list creates survivorship bias in historical tests. Supply a historical universe for research-grade work.
        - Free provider data can be delayed, missing or restated. Validate candidates against company filings.
        - The backtest includes a configurable turnover cost but not taxes, spread variation, market impact or capacity.
        - Percentile scores are relative to the loaded universe. Changing the universe can change every score.
        - Demo securities and prices are fictional. Live scores are research prioritisation, never an instruction to trade.
        """
    )
