from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any

__all__ = [
    "parse_wavelength",
    "select_spectral_columns",
    "coerce_spectral_matrix",
    "decide_domain_by_signature",
    "prepare_for_plot_legacy",
]

def parse_wavelength(colname: Any) -> float | None:
    try:
        s = str(colname).strip().replace(",", ".")
        v = float(s)
        return v if 400.0 <= v <= 2500.0 else None
    except Exception:
        return None

def select_spectral_columns(df: pd.DataFrame) -> Tuple[List[str], List[float]]:
    pairs: List[Tuple[str, float]] = []
    for c in df.columns:
        w = parse_wavelength(c)
        if w is not None:
            pairs.append((c, w))
    if not pairs:
        return [], []
    pairs.sort(key=lambda t: t[1])
    return [c for c, _ in pairs], [float(w) for _, w in pairs]

def coerce_spectral_matrix(df_num: pd.DataFrame) -> np.ndarray:
    """
    Conversão vetorizada: evita fragmentação de DataFrame.
    - troca vírgula por ponto em TODAS as células (apenas uma vez)
    - to_numeric em bloco
    - imputação leve por mediana (preserva domínio)
    """
    # 1) tudo para string uma única vez e troca vírgula por ponto
    s = df_num.astype("string").apply(lambda col: col.str.replace(",", ".", regex=False))
    # 2) numérico coluna-a-coluna, mas construindo um novo DF de uma vez (sem .insert)
    cols = {c: pd.to_numeric(s[c], errors="coerce") for c in s.columns}
    dfc = pd.DataFrame(cols, index=df_num.index)
    X = dfc.to_numpy(dtype=float, copy=True)

    X[np.isinf(X)] = np.nan
    # remove colunas 100% NaN
    keep = ~np.all(np.isnan(X), axis=0)
    if keep.ndim == 1:
        X = X[:, keep]
    if X.size == 0:
        return X
    # imputação leve
    col_med = np.nanmedian(X, axis=0)
    r, c = np.where(np.isnan(X))
    if r.size:
        X[r, c] = col_med[c]
    return X

def _mask(wls: List[float], center: float, half_width: float) -> np.ndarray:
    w = np.asarray(wls, dtype=float)
    return (w >= center - half_width) & (w <= center + half_width)

def decide_domain_by_signature(X: np.ndarray, wavelengths: List[float]) -> Dict[str, Any]:
    if X.size == 0 or not wavelengths:
        return {"domain": "reflectance", "reason": "empty"}

    mean_curve = np.nanmean(X, axis=0)
    m1450 = _mask(wavelengths, 1450.0, 30.0)
    m1200 = _mask(wavelengths, 1200.0, 50.0)

    if np.any(m1450) and np.any(m1200):
        a = float(np.nanmean(mean_curve[m1450]))
        b = float(np.nanmean(mean_curve[m1200]))
        return {
            "domain": "absorbance" if a > b else "reflectance",
            "reason": "water_band",
            "mu_1450": a, "mu_1200": b, "delta": a - b,
        }

    p10, p50, p90 = np.nanpercentile(X, [10, 50, 90])
    return {
        "domain": "reflectance" if p90 > 1.2 or p50 > 0.6 else "absorbance",
        "reason": "percentile", "p10": float(p10), "p50": float(p50), "p90": float(p90),
    }

def prepare_for_plot_legacy(df: pd.DataFrame) -> Tuple[np.ndarray, List[float], pd.DataFrame, Dict[str, Any]]:
    spectral_cols, wavelengths = select_spectral_columns(df)
    if not spectral_cols:
        raise ValueError("Não encontrei colunas espectrais (cabeçalhos numéricos 400–2500 nm).")

    X = coerce_spectral_matrix(df[spectral_cols])
    info = decide_domain_by_signature(X, wavelengths)
    domain = info.get("domain", "reflectance")

    Z = np.array(X, dtype=float, copy=True)
    # marca zeros/negativos como NaN (antes do log)
    bad = ~np.isfinite(Z) | (Z <= 0)
    frac_bad = np.nanmean(bad, axis=0)
    keep = frac_bad < 0.40
    if keep.ndim == 1 and keep.size == Z.shape[1]:
        Z = Z[:, keep]
        wavelengths = [w for w, k in zip(wavelengths, keep) if k]
    if Z.size == 0:
        raise ValueError("Todas as bandas espectrais foram descartadas por falta de dados válidos.")

    col_med = np.nanmedian(Z, axis=0)
    r, c = np.where(~np.isfinite(Z) | (Z <= 0))
    if r.size:
        Z[r, c] = col_med[c]

    if domain == "reflectance":
        Z = np.clip(Z, 1e-3, None)
        A = -np.log10(Z)
        out = A
    else:
        out = Z

    debug = {"domain": domain, "removed_cols_ratio": float(np.mean(~keep)) if keep.size else 0.0, **info}
    y_df = df.drop(columns=spectral_cols).copy()
    return out, wavelengths, y_df, debug
