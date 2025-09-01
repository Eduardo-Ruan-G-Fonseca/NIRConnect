from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["detect_task_from_y", "is_numeric_like"]

_NUM_KINDS = set("biufc")


def is_numeric_like(y: np.ndarray) -> bool:
    y = np.asarray(y, dtype=object)
    if y.dtype.kind in _NUM_KINDS:
        return True
    s = pd.Series(y.ravel(), dtype="object")
    return pd.to_numeric(s, errors="coerce").notna().mean() > 0.9


def detect_task_from_y(y: np.ndarray, req_mode: str | None) -> str:
    if isinstance(req_mode, str):
        low = req_mode.casefold()
        if "classifica" in low or "pls-da" in low:
            return "classification"
        if "regress" in low or "plsr" in low:
            pass
    if not is_numeric_like(y):
        return "classification"
    yy = np.asarray(y).ravel()
    u = pd.unique(yy[pd.notna(yy)])
    if 2 <= u.size <= min(10, max(2, yy.size // 10)):
        return "classification"
    return "regression"

