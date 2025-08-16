import numpy as np
from collections.abc import Iterator
from typing import Callable, Dict, Any, List, Optional
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, KFold, ShuffleSplit, StratifiedShuffleSplit, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def make_cv(y, method: str | None, n_splits: int | None):
    """Create a cross-validation splitter based on the target ``y``.

    Parameters
    ----------
    y : array-like
        Target values used only to infer class balance when ``method`` is not
        Leave-One-Out.
    method : str or None
        Either ``"LOO"`` for Leave-One-Out or any other value/``None`` to use
        ``StratifiedKFold``. The latter automatically caps ``n_splits`` by the
        smallest class size to avoid empty folds.
    n_splits : int or None
        Desired number of splits for ``StratifiedKFold``. When ``None`` defaults
        to 5. The value is always clamped to ``[2, min_class]``.

    Returns
    -------
    cv : splitter instance
        The configured cross-validation splitter.
    meta : dict
        Metadata describing the effective validation strategy and number of
        splits, useful for including in API responses.
    """

    y = np.asarray(y)

    if method == "LOO":
        cv = LeaveOneOut()
        return cv, {"validation": "LOO", "splits": len(y)}

    labels, counts = np.unique(y, return_counts=True)
    min_class = int(counts.min()) if counts.size else 1
    k = max(2, min(n_splits or 5, min_class))
    try:
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        next(cv.split(np.zeros_like(y), y))
        val_name = "StratifiedKFold"
    except Exception:
        cv = KFold(n_splits=k, shuffle=True, random_state=42)
        val_name = "KFold"
    return cv, {"validation": val_name, "splits": k}


def safe_n_components(n_req: int, n_samples: int, n_features: int) -> int:
    """Garante 1 ≤ n_components ≤ min(n_samples-1, n_features-1)."""
    hard_max = max(1, min(n_samples - 1, n_features - 1))
    return max(1, min(int(n_req or 1), hard_max))



def build_cv(
    method: str,
    y: np.ndarray,
    classification: bool,
    params: dict | None,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield train/test indices for the requested CV method.

    The helper caps ``n_splits`` by the smallest class frequency when
    ``classification`` is True and falls back to a simple ``KFold`` when the
    split would be degenerate (e.g. single-class folds).
    """

    method = (method or "").upper()
    params = params or {}

    if method in ("LOO", "LEAVE-ONE-OUT"):
        loo = LeaveOneOut()
        for tr, te in loo.split(np.zeros((len(y), 1)), y if classification else None):
            yield tr, te
        return

    strat_names = {
        "SKF",
        "STRATIFIEDK",
        "STRATIFIEDK_FOLD",
        "STRATIFIEDKFOLD",
        "STRATIFIEDK-FOLD",
        "STRATIFIED",
    }

    if classification and (method in strat_names or method == "KFold" or not method):
        folds = int(params.get("n_splits", 5))
        _, counts = np.unique(y, return_counts=True)
        max_splits = int(counts.min())
        if max_splits < 2:
            cv = KFold(n_splits=min(3, len(y)))
            for tr, te in cv.split(np.zeros((len(y), 1))):
                yield tr, te
            return
        n_splits = int(min(max(2, folds), max_splits))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for tr, te in cv.split(np.zeros((len(y), 1)), y):
            yield tr, te
        return

    folds = int(params.get("n_splits", 5))
    n = int(len(y))
    n_splits = int(min(max(2, folds), max(2, n)))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr, te in cv.split(np.zeros((len(y), 1))):
        yield tr, te


def evaluate_plsda_multiclass(model_factory: Callable[[], Any], X: np.ndarray, y: np.ndarray, method: str = "LOO", params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Perform cross-validation for multi-class PLS-DA models.

    Parameters
    ----------
    model_factory: callable
        Function returning an object with ``fit`` and ``predict`` methods.
    X: ndarray
        Feature matrix.
    y: ndarray
        Target labels (strings or numbers).
    method: str
        Validation method: ``LOO``, ``KFold`` or ``Holdout``.
    params: dict, optional
        Additional parameters for the validation splitter.
    """

    y = np.asarray(y)
    preds: List[str] = []
    trues: List[str] = []

    if method == "LOO":
        cv = LeaveOneOut()
        for tr, te in cv.split(X):
            m = model_factory()
            m.fit(X[tr], y[tr])
            p = m.predict(X[te])
            preds.extend(p.tolist())
            trues.extend(y[te].tolist())
    elif method == "KFold":
        n_splits = int((params or {}).get("n_splits", 5))
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        for tr, te in cv.split(X):
            m = model_factory()
            m.fit(X[tr], y[tr])
            p = m.predict(X[te])
            preds.extend(p.tolist())
            trues.extend(y[te].tolist())
    elif method == "Holdout":
        test_size = float((params or {}).get("test_size", 0.3))
        tr, te = train_test_split(np.arange(len(X)), test_size=test_size, random_state=42, stratify=y)
        m = model_factory()
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        preds = p.tolist()
        trues = y[te].tolist()
    else:
        raise ValueError("validation_method inválido")

    labels = np.unique(y)
    acc = accuracy_score(trues, preds)
    prec = precision_score(trues, preds, labels=labels, average="macro", zero_division=0)
    rec = recall_score(trues, preds, labels=labels, average="macro", zero_division=0)
    f1 = f1_score(trues, preds, labels=labels, average="macro", zero_division=0)
    cm = confusion_matrix(trues, preds, labels=labels).tolist()

    return {
        "Accuracy": float(acc),
        "MacroPrecision": float(prec),
        "MacroRecall": float(rec),
        "MacroF1": float(f1),
        "confusion_matrix": cm,
    }
