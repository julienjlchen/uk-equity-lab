import pytest

from uk_equity_lab.data import _normalise_debt_to_equity, _normalise_dividend_yield


def test_provider_units_are_normalised():
    assert _normalise_dividend_yield(4.79) == pytest.approx(0.0479)
    assert _normalise_dividend_yield(0.0479) == pytest.approx(0.0479)
    assert _normalise_debt_to_equity(95.1) == pytest.approx(0.951)
    assert _normalise_debt_to_equity(0.951) == pytest.approx(0.951)
