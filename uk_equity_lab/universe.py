"""Editable starter universe for London-listed shares."""

from __future__ import annotations

from typing import List


# An intentionally small, liquid starter set. Constituents change over time, so
# users should upload their own point-in-time universe for research-grade tests.
DEFAULT_UK_TICKERS = [
    "AAL.L",
    "ABF.L",
    "AZN.L",
    "BA.L",
    "BARC.L",
    "BATS.L",
    "BP.L",
    "BT-A.L",
    "CPG.L",
    "DGE.L",
    "EXPN.L",
    "GSK.L",
    "HLMA.L",
    "HSBA.L",
    "IMB.L",
    "LAND.L",
    "LGEN.L",
    "LLOY.L",
    "LSEG.L",
    "MKS.L",
    "NG.L",
    "NWG.L",
    "PRU.L",
    "REL.L",
    "RIO.L",
    "RKT.L",
    "RR.L",
    "SBRY.L",
    "SHEL.L",
    "SMT.L",
    "SN.L",
    "SSE.L",
    "STAN.L",
    "TSCO.L",
    "ULVR.L",
    "VOD.L",
    "WPP.L",
]


def clean_tickers(raw: str) -> List[str]:
    """Parse comma/newline separated symbols and apply the LSE suffix."""

    tokens = raw.replace(",", " ").replace(";", " ").split()
    result = []
    for token in tokens:
        ticker = token.strip().upper()
        if not ticker:
            continue
        if ticker.startswith("^") or ticker.endswith(".L"):
            normalised = ticker
        else:
            normalised = "{}.L".format(ticker)
        if normalised not in result:
            result.append(normalised)
    return result
