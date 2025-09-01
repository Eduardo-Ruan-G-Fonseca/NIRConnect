from __future__ import annotations

import re
import numpy as np
import pandas as pd
from typing import Optional, Tuple

__all__ = ["normalize_series_for_target", "pick_column_ci", "load_target_or_fail"]

MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "–", "—"}


def _norm(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s.casefold()


def pick_column_ci(df: pd.DataFrame, target_name: str) -> Optional[str]:
    if not target_name:
        return None
    m = {_norm(c): c for c in df.columns}
    return m.get(_norm(target_name))


def normalize_series_for_target(s: pd.Series) -> pd.Series:
    sc = s.copy()
    if sc.dtype == object or pd.api.types.is_string_dtype(sc):
        sc = sc.astype("string").str.strip()
        sc = sc.mask(sc.fillna("").str.casefold().isin(MISSING_TOKENS))
        # vírgula decimal -> ponto
        cm = sc.str.contains(r"^\s*[-+]?\d+,\d+\s*$", regex=True, na=False)
        sc.loc[cm] = sc.loc[cm].str.replace(",", ".", regex=False)
        as_num = pd.to_numeric(sc, errors="coerce")
        sc = as_num.where(~as_num.isna(), sc)  # tenta número; mantém string se não der
    else:
        sc = pd.to_numeric(sc, errors="coerce")
    return sc


def load_target_or_fail(ds: dict, target_name: str) -> Tuple[np.ndarray, None]:
    ydf = ds.get("y_df")
    if ydf is None or not isinstance(ydf, pd.DataFrame):
        raise ValueError(
            "Dataset não possui y_df persistido. Refaça o upload em /columns."
        )
    col = pick_column_ci(ydf, target_name)
    if col is None:
        raise ValueError(
            f"Coluna-alvo '{target_name}' não encontrada. Disponíveis: {list(ydf.columns)}"
        )
    s = normalize_series_for_target(ydf[col])
    if s.isna().all():
        raise ValueError(
            f"A coluna-alvo '{col}' está vazia ou inválida (após normalização)."
        )
    return s.to_numpy(), None

