from uk_equity_lab.universe import clean_tickers


def test_clean_tickers_adds_london_suffix_and_deduplicates():
    assert clean_tickers("azn, BP.L\nazn") == ["AZN.L", "BP.L"]
