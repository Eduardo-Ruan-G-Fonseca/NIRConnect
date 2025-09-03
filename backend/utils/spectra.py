from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any

__all__ = [
    "parse_wavelength",
    "select_spectral_columns",
    "coerce_spectral_matrix",
    "autoscale_to_legacy_domain",
    "extract_spectra_like_legacy",
]

def parse_wavelength(colname: Any) -> float | None:
    """
    Converte o nome da coluna em float aceitando vírgula decimal.
    Considera válido apenas intervalo NIR típico [400, 2500].
    """
    try:
        s = str(colname).strip().replace(",", ".")
        v = float(s)
        if 400.0 <= v <= 2500.0:
            return v
        return None
    except Exception:
        return None

def select_spectral_columns(df: pd.DataFrame) -> Tuple[List[str], List[float]]:
    """
    Igual ao projeto antigo: escolhe SOMENTE colunas cujo cabeçalho vira float (wavelength).
    Ordena por comprimento de onda crescente.
    """
    cols_wl: List[Tuple[str, float]] = []
    for c in df.columns:
        w = parse_wavelength(c)
        if w is not None:
            cols_wl.append((c, w))
    if not cols_wl:
        return [], []
    # ordena por lambda crescente mantendo par (nome original, valor float)
    cols_wl.sort(key=lambda t: t[1])
    cols_sorted = [c for c, _ in cols_wl]
    wls_sorted = [float(w) for _, w in cols_wl]
    return cols_sorted, wls_sorted

def coerce_spectral_matrix(df_num: pd.DataFrame) -> np.ndarray:
    """
    Converte células para float aceitando vírgula decimal nas células.
    Mantém forma (n_amostras, n_wavelengths).
    """
    out = pd.DataFrame(index=df_num.index)
    for c in df_num.columns:
        s = df_num[c]
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            s = s.astype("string").str.replace(",", ".", regex=False)
        out[c] = pd.to_numeric(s, errors="coerce")
    X = out.to_numpy(dtype=float, copy=True)
    # Sanitização mínima (sem distorcer o domínio do gráfico)
    X[np.isinf(X)] = np.nan
    # remove colunas 100% NaN para não quebrar
    keep = ~np.all(np.isnan(X), axis=0)
    X = X[:, keep] if keep.ndim == 1 else X
    # imputação leve por coluna (mediana é mais robusta e mantém escala)
    if X.size:
        col_med = np.nanmedian(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_med, inds[1])
    return X

def autoscale_to_legacy_domain(X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Emula o legado para exibir curvas ~[0.05, 0.6] com pico ~1450 nm:
    - Se parecer reflectância em % → divide por 100 (0..1)
    - Se intensidades altas → divide por 10000 (caso comum de instrumentos)
    - Se estiver em reflectância (0..~1.2) → retorna A = -log10(R)
    Caso contrário, mantém em reflectância (já coerente).
    """
    Z = np.array(X, dtype=float, copy=True)
    Z[np.isinf(Z)] = np.nan
    # heurística de escala
    p10 = np.nanpercentile(Z, 10)
    p50 = np.nanpercentile(Z, 50)
    p90 = np.nanpercentile(Z, 90)

    scale_info: Dict[str, Any] = {"domain": "unknown", "pre": {"p10": float(p10), "p50": float(p50), "p90": float(p90)}}

    # % reflectance?
    if p90 > 1.5 and p90 <= 120.0 and p10 >= 0.0:
        Z = Z / 100.0
        scale_info["hint"] = "percent_to_unit"
    # raw counts muito altos (ex.: 0..10000)
    elif p90 > 1000.0:
        Z = Z / 10000.0
        scale_info["hint"] = "counts_to_unit"

    # clamp leve para evitar log de <=0
    Z = np.clip(Z, 1e-6, None)

    # Se agora parece reflectância (0..≈1.2), use absorbância A = -log10(R)
    med = float(np.nanmedian(Z))
    if 0.02 <= med <= 1.2:
        A = -np.log10(Z)
        scale_info["domain"] = "absorbance"
        return A, scale_info
    else:
        scale_info["domain"] = "reflectance"
        return Z, scale_info

def extract_spectra_like_legacy(df: pd.DataFrame) -> Tuple[np.ndarray, List[float], pd.DataFrame, Dict[str, Any]]:
    """
    Pipeline legado para o gráfico:
      1) selecionar colunas espectrais pelo cabeçalho numérico 400..2500
      2) ordenar por lambda crescente
      3) converter células aceitando vírgula decimal
      4) autoscale + absorbância se aplicável
    Retorna: (X_plot, wavelengths, y_df, debug_info)
    """
    spectral_cols, wavelengths = select_spectral_columns(df)
    if not spectral_cols:
        raise ValueError("Não encontrei colunas espectrais (cabeçalhos numéricos 400–2500 nm).")
    X_raw = coerce_spectral_matrix(df[spectral_cols])
    X_plot, info = autoscale_to_legacy_domain(X_raw)
    y_df = df.drop(columns=spectral_cols).copy()
    debug = {"spectral_cols": spectral_cols, "wavelengths": wavelengths, **info}
    return X_plot, wavelengths, y_df, debug
