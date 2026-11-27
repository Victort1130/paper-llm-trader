import math

import numpy as np
import pandas as pd
import pytest

from papertrader.data.indicators import compute_indicators, realized_vol, rsi, sma


def _s(values):
    return pd.Series(values, index=pd.bdate_range("2024-01-01", periods=len(values)), dtype=float)


def test_sma_and_short_history():
    assert sma(_s([1, 2, 3, 4]), 2) == 3.5
    assert sma(_s([1]), 2) is None


def test_rsi_extremes():
    assert rsi(_s(range(1, 30))) == 100.0
    assert rsi(_s(range(30, 1, -1))) == 0.0
    assert rsi(_s([5] * 30)) == 50.0


def test_constant_growth_has_zero_vol():
    s = _s(100 * 1.01 ** np.arange(40))
    assert realized_vol(s, 20) == pytest.approx(0.0, abs=1e-12)


def test_known_returns():
    s = _s(list(range(100, 160)))
    ind = compute_indicators(pd.DataFrame({"close": s}))
    assert ind["ret_5d"] == pytest.approx(159 / 154 - 1)
    assert ind["ret_20d"] == pytest.approx(159 / 139 - 1)
    assert ind["close_vs_sma20"] == pytest.approx(159 / np.mean(range(140, 160)) - 1)
    assert math.isfinite(ind["vol_20d"])
