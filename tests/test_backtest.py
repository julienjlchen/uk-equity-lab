from copy import deepcopy

import numpy as np
import pandas as pd

from uk_equity_lab.backtest import run_backtest
from uk_equity_lab.config import DEFAULT_CONFIG


def test_backtest_forms_then_executes_on_next_session():
    dates = pd.bdate_range("2021-01-01", periods=180)
    close = pd.DataFrame(
        {
            "A.L": 10 * np.cumprod(np.repeat(1.002, len(dates))),
            "B.L": 10 * np.cumprod(np.repeat(1.000, len(dates))),
            "C.L": 10 * np.cumprod(np.repeat(0.999, len(dates))),
        },
        index=dates,
    )
    config = deepcopy(DEFAULT_CONFIG)
    config["backtest"]["minimum_history_days"] = 40
    config["methodology"]["scoring"]["minimum_coverage"] = 0.0
    result = run_backtest(
        close,
        config,
        start=dates[45],
        end=dates[-1],
        top_n=1,
        frequency="M",
        transaction_cost_bps=10,
    )
    assert not result.rebalances.empty
    assert (result.rebalances["execution_date"] > result.rebalances["formation_date"]).all()
    assert (result.holdings.groupby("formation_date").size() == 1).all()
    assert result.equity.index.min() == result.rebalances["execution_date"].min()
    assert result.metrics["total_cost"] > 0
