import numpy as np
from sklearn.model_selection import LeaveOneOut, StratifiedKFold, KFold


def build_cv_meta(method: str, params: dict, y):
    """Constrói o objeto de validação e retorna (cv, meta).

    Parameters
    ----------
    method : str
        Nome do método solicitado (ex.: ``"LOO"``, ``"StratifiedKFold"``).
    params : dict
        Parâmetros adicionais para o construtor da validação. Atualmente
        apenas ``n_splits`` é utilizado.
    y : array-like
        Vetor de respostas/alvos. Será convertido de forma segura para um
        ``numpy.ndarray`` 1D.

    Returns
    -------
    tuple
        ``(cv, meta)`` onde ``cv`` é o objeto de validação pronto para uso e
        ``meta`` é um ``dict`` com informações sobre o método efetivamente
        utilizado, número de divisões e quantidade de amostras.
    """

    # Garantir vetor 1D com tamanho conhecido (evita "len() of unsized object")
    y_arr = np.asarray(y).ravel()
    n_samples = int(y_arr.shape[0])

    method = (method or "").strip().upper()
    params = params or {}

    if method == "LOO":
        cv = LeaveOneOut()
        splits = n_samples
        meta_method = "LOO"

    elif method in {"SKF", "STRATIFIEDK", "STRATIFIEDKFOLD", "STRATIFIED"}:
        n_splits = int(params.get("n_splits", 5)) if params else 5
        classes, counts = np.unique(y_arr, return_counts=True)
        # StratifiedKFold requer pelo menos duas classes e que cada uma tenha
        # quantidade mínima de amostras para todas as dobras
        if classes.size >= 2 and counts.min(initial=0) >= n_splits and n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = n_splits
            meta_method = "StratifiedKFold"
        else:
            # Fallback seguro para KFold quando estratificação não é possível
            safe_splits = max(2, min(n_splits, n_samples))
            cv = KFold(n_splits=safe_splits, shuffle=True, random_state=42)
            splits = safe_splits
            meta_method = "KFold(fallback)"

    else:
        n_splits = int(params.get("n_splits", 5)) if params else 5
        safe_splits = max(2, min(n_splits, n_samples))
        cv = KFold(n_splits=safe_splits, shuffle=True, random_state=42)
        splits = safe_splits
        meta_method = "KFold"

    meta = {"method": meta_method, "splits": splits, "n_samples": n_samples}
    return cv, meta

