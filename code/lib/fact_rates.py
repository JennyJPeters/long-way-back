#!/usr/bin/env python3
from __future__ import annotations
import numpy as np
import pandas as pd


def clustered_rate(sub: pd.DataFrame, cond) -> tuple[float, float]:
    """Weighted share of `cond` in `sub` and its SE clustered by cpsidp; returns (p_hat, se)."""
    if len(sub) == 0:
        return np.nan, np.nan
    w = sub['lnkfw1mwt'].to_numpy(float)
    y = np.asarray(cond, dtype=float)
    W = w.sum()
    if W <= 0:
        return np.nan, np.nan
    p = float((w * y).sum() / W)
    infl = (w / W) * (y - p)
    g = pd.Series(infl, index=sub.index).groupby(sub['cpsidp'].to_numpy()).sum().to_numpy()
    G = g.size
    if G < 2:
        return p, np.nan
    var = (G / (G - 1.0)) * np.square(g).sum()     # small-sample scaling
    return p, float(np.sqrt(max(var, 0.0)))
