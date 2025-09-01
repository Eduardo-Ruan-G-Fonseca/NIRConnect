from __future__ import annotations
import re
import numpy as np
import pandas as pd
from typing import Optional, Tuple

MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "–", "—"}

def _sanitize_colname(name: str) -> str:
    # casefold + remove espaços extras + trocar separadores estranhos
    s = (name or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s.casefold()

def normalize_series_for_target(s: pd.Series) -> pd.Series:
    """
    - tira espaços
    - normaliza missing tokens
    - troca vírgula por ponto em strings numéricas
    - preserva strings para classificação; números viram float
    """
    if s is None:
        return s

    # trabalha em cópia
    sc = s.copy()

    # strings: strip, vírgula->ponto
    if sc.dtype == object or pd.api.types.is_string_dtype(sc):
        sc = sc.astype("string").str.strip()

        # strings vazias / tokens de missing -> <NA>
        sc = sc.mask(sc.fillna("").str.casefold().isin(MISSING_TOKENS))

        # trocar vírgula decimal por ponto (onde aplicável)
        comma_as_decimal = sc.str.contains(r"^\s*[-+]?\d+,\d+\s*$", regex=True, na=False)
        sc.loc[comma_as_decimal] = sc.loc[comma_as_decimal].str.replace(",", ".", regex=False)

        # tentar converter para número onde possível, senão mantém string
        sc_num = pd.to_numeric(sc, errors="coerce")
        # onde deu número, usa número; onde não deu, mantém string (para classificação)
        sc = sc_num.where(~sc_num.isna(), sc)

    else:
        # numérico: apenas garantir float e NaNs corretos
        sc = pd.to_numeric(sc, errors="coerce")

    return sc

def pick_column_ci(df: pd.DataFrame, target_name: str) -> Optional[str]:
    """Retorna o nome real da coluna no DF, fazendo match case-insensitive e normalizado."""
    if target_name is None or target_name == "":
        return None
    wanted = _sanitize_colname(target_name)
    mapping = { _sanitize_colname(c): c for c in df.columns }
    return mapping.get(wanted)

def load_target_or_fail(
    ds: dict, target_name: str
) -> Tuple[np.ndarray, Optional[list]]:
    """
    Lê y de ds['y_df'] pela coluna target_name (robusto).
    - Normaliza série (vide normalize_series_for_target)
    - Retorna:
        y_raw (np.ndarray) e classes (lista) quando categórico, senão None
    - Levanta ValueError com mensagem clara se não encontrar coluna ou se ficar toda NaN.
    """
    ydf = ds.get("y_df")
    if ydf is None or not isinstance(ydf, pd.DataFrame):
        raise ValueError("Dataset não possui y_df persistido. Refaça o upload em /columns.")

    col = pick_column_ci(ydf, target_name)
    if col is None:
        raise ValueError(f"Coluna-alvo '{target_name}' não encontrada. Disponíveis: {list(ydf.columns)}")

    s = normalize_series_for_target(ydf[col])

    # Se ficou tudo NA/NaN → erro direto (melhor do que alinhar e zerar amostras)
    if s.isna().all():
        raise ValueError(f"A coluna-alvo '{col}' está vazia ou inválida (após normalização).")

    # retorna np.ndarray (pode conter strings e números mistos; o caller decide o tratamento)
    y = s.to_numpy()
    # classes (apenas se claramente categórico: dtype string ou object sem conseguir virar tudo número)
    classes = None
    # heurística: se não é estritamente numérico depois da conversão, trate como categórico
    if pd.api.types.is_string_dtype(s) or s.dtype == object:
        # não extraímos classes aqui (o PLS-DA pipeline via LabelEncoder fará isso)
        # devolvemos None e deixamos sanitize_y cuidar
        classes = None

    return y, classes
