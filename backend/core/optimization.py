import time
from typing import Any, Dict, List, Callable

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    r2_score,
    cohen_kappa_score,
)
from sklearn.model_selection import cross_val_predict

from .preprocessing import apply_methods
from .validation import make_cv
from .pls import fit_plsda_multiclass
from utils.sanitize import sanitize_X, sanitize_y, limit_n_components


# ---------------------------------------------------------------------------
# Helper models and preprocessing
# ---------------------------------------------------------------------------

def preprocess(X: np.ndarray, method: str | None, range_nm: tuple[int, int] | None = None) -> np.ndarray:
    """Apply optional preprocessing to ``X``.

    Parameters
    ----------
    X : ndarray
        Feature matrix.
    method : str or None
        Preprocessing method name. When ``None`` or ``"none"`` the input is
        returned unchanged.
    range_nm : tuple[int, int] or None
        Optional wavelength range (ignored here but kept for API compatibility).
    """

    methods = [] if not method or method == "none" else [method]
    Xp = apply_methods(X, methods=methods)
    return Xp


class _PLSDAEstimator(BaseEstimator):
    """Thin wrapper around :func:`fit_plsda_multiclass` for ``cross_val_predict``."""

    def __init__(self, n_components: int):
        self.n_components = int(n_components)

    def fit(self, X, y):
        self.model_ = fit_plsda_multiclass(X, y, n_components=self.n_components)
        return self

    def predict(self, X):
        return self.model_.predict(X)


def make_pls_da(n_components: int) -> BaseEstimator:
    return _PLSDAEstimator(n_components=n_components)


def make_pls_reg(n_components: int) -> PLSRegression:
    return PLSRegression(n_components=n_components)


# ---------------------------------------------------------------------------
# Legacy optimisation helper retained for compatibility
# ---------------------------------------------------------------------------

def optimize_nir(
    X: np.ndarray,
    y: np.ndarray,
    wl: np.ndarray | None,
    classification: bool,
    n_components_list: List[int] | None = None,
    n_intervals: int = 10,
) -> List[Dict[str, Any]]:
    """Minimal wrapper kept for backward compatibility with existing tests."""

    from .pls import train_pls  # local import to avoid cycle

    n_components_list = n_components_list or [2, 3, 4, 5]
    results: List[Dict[str, Any]] = []
    if wl is None:
        for nc in n_components_list:
            res = train_pls(X, y, X, y, n_components=nc, classification=classification)
            results.append({"range": None, "n_components": nc, "metrics": res["metrics"]})
        return sorted(
            results,
            key=lambda r: r["metrics"].get("RMSE", 1e9)
            if not classification
            else -r["metrics"].get("Accuracy", 0),
        )

    min_wl, max_wl = wl.min(), wl.max()
    points = np.linspace(min_wl, max_wl, n_intervals + 1)
    intervals = [(points[i], points[i + 1]) for i in range(len(points) - 1)]
    for (wmin, wmax) in intervals:
        mask = (wl >= wmin) & (wl <= wmax)
        if mask.sum() < 3:
            continue
        Xw = X[:, mask]
        for nc in n_components_list:
            res = train_pls(Xw, y, Xw, y, n_components=nc, classification=classification)
            score = (
                res["metrics"].get("RMSE", 1e9)
                if not classification
                else -res["metrics"].get("Accuracy", 0)
            )
            results.append(
                {
                    "range": (float(wmin), float(wmax)),
                    "n_components": nc,
                    "metrics": res["metrics"],
                    "score": score,
                }
            )
    return sorted(results, key=lambda r: r["score"])


# ---------------------------------------------------------------------------
# New grid search implementation
# ---------------------------------------------------------------------------

def optimize_model_grid(
    X,
    y,
    mode,
    preprocessors,
    n_components_max,
    validation_method,
    n_splits,
    wavelength_range,
    logger,
    time_budget_s=None,
):
    """Grid-search over preprocessing methods and number of PLS components."""
    task = "classification" if mode in (
        "classification",
        "pls-da",
        "PLS-DA",
        "Classificação (PLS-DA)",
    ) else "regression"

    X = sanitize_X(X)
    y = sanitize_y(y, task)
    if X is None or X.size == 0:
        return {"status": "error", "message": "Matriz X vazia após sanitização."}

    cv, cv_meta = make_cv(y, validation_method, n_splits)
    labels_all = sorted(list(np.unique(y)))
    n_feat, n_samp = X.shape[1], X.shape[0]
    max_nc_safe = min(n_feat // 4, n_samp - 1, 50)
    max_nc = min(n_components_max or max_nc_safe, max_nc_safe)
    max_nc = max(max_nc, 1)

    results, curves_map = [], {}
    t0 = time.time()

    for prep in (preprocessors or [None]):
        Xp = preprocess(X, method=prep, range_nm=wavelength_range)
        for nc in range(1, max_nc + 1):
            ncomp = limit_n_components(nc, Xp)
            if ncomp < 1:
                continue
            if mode == "classification":
                est = make_pls_da(n_components=ncomp)
                y_pred = cross_val_predict(est, Xp, y, cv=cv)
                acc = float(accuracy_score(y, y_pred))
                f1m = float(
                    f1_score(
                        y, y_pred, average="macro", labels=labels_all, zero_division=0
                    )
                )
                rep = classification_report(
                    y,
                    y_pred,
                    labels=labels_all,
                    output_dict=True,
                    zero_division=0,
                )
                per_class = {}
                for lbl in labels_all:
                    k = str(lbl)
                    d = rep.get(k, {})
                    per_class[k] = {
                        "precision": float(d.get("precision", 0.0)),
                        "recall": float(d.get("recall", 0.0)),
                        "f1": float(d.get("f1-score", 0.0)),
                        "support": int(d.get("support", (y == lbl).sum())),
                    }
                cm = confusion_matrix(y, y_pred, labels=labels_all).tolist()
                kappa = float(cohen_kappa_score(y, y_pred, labels=labels_all)) if len(labels_all) > 1 else 0.0
                met_full = {
                    "Accuracy": round(acc, 4),
                    "MacroF1": round(f1m, 4),
                    "Kappa": round(kappa, 4),
                    "per_class": per_class,
                    "confusion_matrix": cm,
                }
                curve_val = met_full["MacroF1"]
                met_curve = {"Accuracy": met_full["Accuracy"], "MacroF1": met_full["MacroF1"]}
            else:
                est = make_pls_reg(n_components=ncomp)
                y_cv = cross_val_predict(est, Xp, y, cv=cv)
                rmse = float(np.sqrt(mean_squared_error(y, y_cv)))
                mae = float(np.mean(np.abs(y - y_cv)))
                r2 = float(r2_score(y, y_cv))
                met_full = {
                    "RMSECV": round(rmse, 6),
                    "R2CV": round(r2, 6),
                    "MAECV": round(mae, 6),
                }
                curve_val = met_full["RMSECV"]
                met_curve = {
                    "RMSECV": met_full["RMSECV"],
                    "R2CV": met_full["R2CV"],
                    "MAECV": met_full["MAECV"],
                }

            results.append(
                {
                    "preprocess": prep or "none",
                    "n_components": ncomp,
                    "metrics": met_full,
                    **met_curve,
                }
            )
            curves_map.setdefault(prep or "none", []).append(
                {
                    "n_components": ncomp,
                    **({"MacroF1": curve_val} if mode == "classification" else {"RMSECV": curve_val}),
                }
            )
            if logger:
                logger.info(f"[grid] prep={prep or 'none'} nc={nc} metrics={met_full}")
            if time_budget_s and time.time() - t0 > time_budget_s:
                break
        if time_budget_s and time.time() - t0 > time_budget_s:
            break

    best = (
        max(results, key=lambda r: r["MacroF1"])
        if mode == "classification"
        else min(results, key=lambda r: r["RMSECV"])
    )

    curves = []
    for k, pts in curves_map.items():
        curves.append({"preprocess": k, "points": sorted(pts, key=lambda p: p["n_components"])})

    return {
        "results": results,
        "best": {
            "preprocess": best["preprocess"],
            "n_components": best["n_components"],
            "metrics": best["metrics"],
        },
        "curves": curves,
        "validation": cv_meta,
        "range_used": list(wavelength_range) if wavelength_range else None,
        "labels_all": labels_all,
    }

