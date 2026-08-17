import pytest
import pandas as pd

from uk_equity_lab.paper import PaperLedger, equal_weight_order_plan


def test_paper_ledger_never_allows_overspend_or_short(tmp_path):
    ledger = PaperLedger(tmp_path / "trades.csv", starting_cash=1000, fee_bps=10)
    ledger.record_trade("AAA.L", "BUY", 10, 20)
    state = ledger.state({"AAA.L": 25})
    assert state.cash == pytest.approx(799.8)
    assert state.total_value == pytest.approx(1049.8)
    with pytest.raises(ValueError, match="short"):
        ledger.record_trade("AAA.L", "SELL", 11, 25)
    with pytest.raises(ValueError, match="cash"):
        ledger.record_trade("AAA.L", "BUY", 100, 20)


def test_equal_weight_plan_respects_whole_shares():
    plan = equal_weight_order_plan(["A.L", "B.L"], {"A.L": 30, "B.L": 40}, 1000)
    assert plan.set_index("ticker").loc["A.L", "quantity"] == 16
    assert plan.set_index("ticker").loc["B.L", "quantity"] == 12
    assert plan["notional"].sum() <= 1000


def test_exported_ledger_can_be_restored(tmp_path):
    original = PaperLedger(tmp_path / "original.csv", starting_cash=1000, fee_bps=10)
    original.record_trade("AAA.L", "BUY", 10, 20)
    exported = tmp_path / "exported.csv"
    original.trades().to_csv(exported, index=False)

    restored = PaperLedger(tmp_path / "restored.csv", starting_cash=1000, fee_bps=10)
    restored.restore(exported)
    assert len(restored.trades()) == 1
    assert restored.state({"AAA.L": 25}).total_value == pytest.approx(1049.8)


def test_restore_rejects_invalid_short_history(tmp_path):
    invalid = pd.DataFrame(
        [
            {
                "id": "bad",
                "timestamp": "2026-01-01T12:00:00Z",
                "ticker": "AAA.L",
                "side": "SELL",
                "quantity": 1,
                "price": 20,
                "fee": 0,
                "note": "invalid",
            }
        ]
    )
    source = tmp_path / "invalid.csv"
    invalid.to_csv(source, index=False)
    ledger = PaperLedger(tmp_path / "target.csv", starting_cash=1000)
    with pytest.raises(ValueError, match="short"):
        ledger.restore(source)
