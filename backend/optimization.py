from __future__ import annotations

import numpy as np
from typing import Dict, Any, Iterable
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error

from utils.task_detect import detect_task_from_y
from utils.sanitize import sanitize_X, sanitize_y, align_X_y, limit_n_components
from validation import build_cv
from pls import make_pls_reg, make_pls_da


def optimize_model_grid(
    X: np.ndarray,
    y: np.ndarray,
    grid: Dict[str, Iterable[Any]],
    mode: str,
    validation_method: str = "KFold",
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Evaluate PLS models over a grid of ``n_components`` values.

    The function is robust against invalid combinations: when an error occurs
    during cross-validation for a specific number of components the exception is
    caught and recorded while the search continues. It supports both regression
    and classification tasks, automatically selecting between PLSR and PLS-DA.
    """

    out: Dict[str, Any] = {"status": "error", "message": "sem avaliação", "history": []}

    task = detect_task_from_y(y, mode)
    X = sanitize_X(X)
    y, classes_ = sanitize_y(y, task)
    X, y, _ = align_X_y(X, y)
    if X.shape[0] == 0:
        return {"status": "error", "message": "Sem amostras após sanitização."}

    cv = build_cv(
        validation_method,
        y=y,
        n_splits=n_splits,
        stratified=(task == "classification"),
    )

    ncomp_list = list(grid.get("n_components", [2]))
    best_score = -np.inf if task == "classification" else np.inf

    for ncomp in ncomp_list:
        scores = []
        try:
            for tr, te in cv.split(X, y):
                Xtr, Xte = X[tr], X[te]
                ytr, yte = y[tr], y[te]
                safe_n = limit_n_components(int(ncomp), Xtr)
                if safe_n < 1:
                    continue

                if task == "classification":
                    model = make_pls_da(
                        n_components=safe_n,
                        n_classes=int(np.unique(y).size),
                    )
                    model.fit(Xtr, ytr.astype(int))
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(Xte)
                        if proba.shape[1] == 2:
                            score = roc_auc_score(yte, proba[:, 1])
                        else:
                            pred = np.argmax(proba, axis=1)
                            score = accuracy_score(yte, pred)
                    else:
                        pred = model.predict(Xte).ravel()
                        if np.unique(y).size <= 2:
                            score = accuracy_score(yte, (pred >= 0.5).astype(int))
                        else:
                            score = accuracy_score(yte, np.rint(pred).astype(int))
                else:
                    model = make_pls_reg(n_components=safe_n).fit(Xtr, ytr)
                    pred = model.predict(Xte).ravel()
                    score = -np.sqrt(mean_squared_error(yte, pred))

                scores.append(float(score))
        except Exception as e:  # pragma: no cover - log errors
            out["history"].append({"n_components": int(ncomp), "error": str(e)})
            continue

        if not scores:
            continue

        mean_score = float(np.mean(scores))
        out["history"].append({"n_components": int(ncomp), "cv_score": mean_score})
        better = mean_score > best_score if task == "classification" else mean_score < best_score
        if better:
            best_score = mean_score
            out.update(
                {
                    "status": "ok",
                    "best_params": {"n_components": int(ncomp)},
                    "best_score": mean_score,
                    "task": task,
                    "classes_": classes_ or [],
                    "validation_method": validation_method,
                }
            )

    if out.get("status") != "ok":
        out["message"] = "Nenhuma combinação válida avaliada."
    return out


__all__ = ["optimize_model_grid"]

