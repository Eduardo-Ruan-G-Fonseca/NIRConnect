from typing import List, Dict, Any, Callable
import os, time
import numpy as np
import pandas as pd

try:  # opcional, só ativa se quiser paralelizar folds
    from joblib import Parallel, delayed  # type: ignore
except Exception:  # pragma: no cover
    Parallel = None
    delayed = None

from .validation import build_cv
from .pls import train_pls
from .preprocessing import apply_methods
from .logger import log_info
from .metrics import regression_metrics, classification_metrics


def _safe_train_pls(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    n_components: int,
    classification: bool,
    extra_kwargs: dict | None,
):
    """Wrapper for :func:`train_pls` ensuring no duplicated arguments.

    - ``n_components`` is always passed explicitly.
    - Any provided kwargs will have ``n_components`` removed.
    - Validation is forced to a simple train/test split (no inner CV).
    """

    if extra_kwargs is None:
        extra_kwargs = {}
    extra_kwargs = dict(extra_kwargs)
    extra_kwargs.pop("n_components", None)

    vm = extra_kwargs.get("validation_method", "none") or "none"
    extra_kwargs["validation_method"] = "none"
    extra_kwargs["validation_params"] = {}

    from core.pls import train_pls  # local import to avoid circular

    # ``train_pls`` in this project expects only the training data. Predictions
    # on ``Xte`` are computed outside this helper.
    return train_pls(
        Xtr,
        ytr,
        n_components=n_components,
        classification=classification,
        **extra_kwargs,
    )

def optimize_nir(
    X: np.ndarray,
    y: np.ndarray,
    wl: np.ndarray | None,
    classification: bool,
    n_components_list: List[int] = [2,3,4,5],
    n_intervals: int = 10,
    method: str = "ipls"
) -> List[Dict[str, Any]]:
    results = []
    if wl is None:
        for nc in n_components_list:
            model, metrics, extra = train_pls(
                X,
                y,
                n_components=nc,
                classification=classification,
            )
            results.append({
                "range": None,
                "n_components": nc,
                "metrics": metrics
            })
        return sorted(results, key=lambda r: r["metrics"].get("RMSE", 1e9) if not classification else -r["metrics"].get("Accuracy", 0))

    min_wl, max_wl = wl.min(), wl.max()
    points = np.linspace(min_wl, max_wl, n_intervals+1)
    intervals = [(points[i], points[i+1]) for i in range(len(points)-1)]

    for (wmin, wmax) in intervals:
        mask = (wl >= wmin) & (wl <= wmax)
        if mask.sum() < 3:
            continue
        Xw = X[:, mask]
        for nc in n_components_list:
            model, metrics, extra = train_pls(
                Xw,
                y,
                n_components=nc,
                classification=classification,
            )

            score = metrics.get("RMSE", 1e9) if not classification else -metrics.get("Accuracy", 0)
            results.append({
                "range": (float(wmin), float(wmax)),
                "n_components": nc,
                "metrics": metrics,
                "score": score
            })
    return sorted(results, key=lambda r: r["score"])


def optimize_model_grid(
    X: np.ndarray,
    y: np.ndarray,
    wl: np.ndarray | None,
    classification: bool,
    preprocess_opts: List[str] | None = None,
    n_components_range: range = range(2, 6),
    n_splits: int = 5,
    validation_method: str | None = None,
    validation_params: Dict[str, Any] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> List[Dict[str, Any]]:
    """Grid-search style optimization over preprocessing and component number."""

    if isinstance(X, pd.DataFrame):
        X = X.values
        log_info(f"X convertido de DataFrame para array: {X.shape}")
    elif not isinstance(X, np.ndarray):
        raise ValueError("X precisa ser DataFrame ou ndarray")

    if isinstance(y, pd.Series):
        y = y.values
        log_info(f"y otimização recebido como Series: {y.shape}")
    elif not isinstance(y, np.ndarray) or y.ndim != 1:
        raise ValueError("y precisa ser uma série unidimensional")

    assert len(X) == len(y), "X e y devem ter o mesmo número de amostras"

    if classification:
        y = y.astype(str)
        labels = sorted(np.unique(y))
    else:
        y = y.astype(float)
        labels = None

    if preprocess_opts is None:
        preprocess_opts = ["none"]

    if validation_method is None:
        validation_method = "StratifiedKFold" if classification else "KFold"
        validation_params = {
            "n_splits": n_splits,
            "shuffle": True,
            "random_state": 42,
        }
    validation_params = validation_params or {}

    splits = list(build_cv(validation_method, y, classification, validation_params))
    cv_splits = int(max(1, len(splits)))
    methods_count = int(len(preprocess_opts) if preprocess_opts else 1)
    total_steps = 0
    done = 0

    cache_Xp: Dict[str, np.ndarray] = {}
    results: List[Dict[str, Any]] = []

    for prep in preprocess_opts:
        if prep not in cache_Xp:
            Xp = apply_methods(X, methods=[prep])
            from .preprocessing import sanitize_X
            Xp, _ = sanitize_X(Xp)
            cache_Xp[prep] = Xp
        else:
            Xp = cache_Xp[prep]

        max_nc = int(min(Xp.shape[1], max(1, Xp.shape[0] - 1)))
        comp_range = [nc for nc in n_components_range if 1 <= nc <= max_nc]
        comp_len = int(len(comp_range))
        total_steps += methods_count * comp_len * cv_splits
        total_steps = int(total_steps)
        if not comp_range:
            done += cv_splits
            if progress_callback:
                progress_callback(int(done), int(max(1, total_steps)))
            continue

        for nc in comp_range:
            try:
                n_jobs = int(os.getenv("NIR_N_JOBS", "1"))

                def eval_one(tr, te):
                    model, _, _ = _safe_train_pls(
                        Xp[tr],
                        y[tr],
                        Xp[te],
                        y[te],
                        n_components=nc,
                        classification=bool(classification),
                        extra_kwargs={},
                    )
                    preds = model.predict(Xp[te])
                    if classification:
                        m = classification_metrics(
                            y[te], np.array(preds).astype(str), labels=labels
                        )
                    else:
                        m = regression_metrics(
                            y[te], np.array(preds, dtype=float)
                        )
                    return m

                if Parallel and delayed and n_jobs > 1:
                    metrics_list = Parallel(n_jobs=n_jobs)(
                        delayed(eval_one)(tr, te) for (tr, te) in splits
                    )
                else:
                    metrics_list = [eval_one(tr, te) for (tr, te) in splits]

                if classification:
                    scores = [m.get("F1") or m.get("MacroF1") or m.get("Accuracy", 0) for m in metrics_list]
                    mean_score = float(np.mean(scores))
                    agg_metrics: Dict[str, Any] = {}
                    for k in metrics_list[0].keys():
                        vals = [m.get(k) for m in metrics_list if isinstance(m.get(k), (int, float))]
                        if vals:
                            agg_metrics[k] = float(np.mean(vals))
                    val_metrics = agg_metrics
                    rmsecv = None
                else:
                    rmse_vals = [m.get("RMSE", np.nan) for m in metrics_list]
                    rmsecv = float(np.mean(rmse_vals))
                    mean_score = -rmsecv
                    agg_metrics: Dict[str, Any] = {}
                    for k in metrics_list[0].keys():
                        vals = [m.get(k) for m in metrics_list if isinstance(m.get(k), (int, float))]
                        if vals:
                            agg_metrics[k] = float(np.mean(vals))
                    val_metrics = agg_metrics

                results.append(
                    {
                        "preprocess": prep,
                        "n_components": nc,
                        "val_metrics": val_metrics,
                        "RMSECV": rmsecv,
                        "validation": {
                            "method": validation_method,
                            "params": validation_params,
                        },
                        "score": mean_score,
                    }
                )

            except Exception as e:
                print(f"[grid] failed prep={prep} n_comp={nc}: {e}", flush=True)
            finally:
                done += cv_splits
                if progress_callback:
                    progress_callback(int(done), int(max(1, total_steps)))

    return sorted(results, key=lambda r: r.get("score", -np.inf), reverse=True)
