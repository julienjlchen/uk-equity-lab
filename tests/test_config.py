from copy import deepcopy

import pytest

from uk_equity_lab.config import DEFAULT_CONFIG, validate_config


def test_weights_are_normalised_without_mutating_input():
    raw = deepcopy(DEFAULT_CONFIG)
    raw["methodology"]["category_weights"] = {
        "fundamentals": 30,
        "valuation": 25,
        "momentum": 30,
        "risk": 15,
    }
    result = validate_config(raw)
    assert sum(result["methodology"]["category_weights"].values()) == pytest.approx(1)
    assert sum(raw["methodology"]["category_weights"].values()) == 100


def test_negative_weight_is_rejected():
    raw = deepcopy(DEFAULT_CONFIG)
    raw["methodology"]["category_weights"]["risk"] = -1
    with pytest.raises(ValueError, match="cannot be negative"):
        validate_config(raw)
