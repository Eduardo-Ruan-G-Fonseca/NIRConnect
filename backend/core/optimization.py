from typing import List, Dict, Any, Callable
import os
import numpy as np
from .validation import build_cv
from .pls import train_pls
from .preprocessing import apply_methods, sanitize_X


def _extract_score(metrics: dict, classification: bool) -> float:
    """Return a suitable optimization score from ``metrics``."""
    import math

    if not isinstance(metrics, dict):
        raise KeyError("metrics not a dict")

    if classification:
        m = {k.lower(): v for k, v in metrics.items()}
        for key in ("f1", "f1_macro", "f1_score", "f1weighted", "f1_weighted"):
            if key in m and math.isfinite(float(m[key])):
                return float(m[key])
        for key in ("balancedaccuracy", "balanced_accuracy", "bal_acc", "bac"):
            if key in m and math.isfinite(float(m[key])):
                return float(m[key])
        for key in ("accuracy", "acc"):
            if key in m and math.isfinite(float(m[key])):
                return float(m[key])
        for key in ("auroc", "auc", "roc_auc", "kappa"):
            if key in m and math.isfinite(float(m[key])):
                return float(m[key])
        raise KeyError(f"No finite classification metric found; got keys={list(metrics.keys())}")

    m = {k.lower(): v for k, v in metrics.items()}
    for key in ("rmse", "mae"):
        if key in m and math.isfinite(float(m[key])):
            return -float(m[key])
    if "r2" in m and math.isfinite(float(m["r2"])):
        return float(m["r2"])
    raise KeyError(f"No finite regression metric found; got keys={list(metrics.keys())}")


def log_info(msg: str):
    try:
        print(msg, flush=True)
    except Exception:
        pass

def optimize_nir(
    X: np.ndarray,
    y: np.ndarray,
    wl: np.ndarray | None,
    classification: bool,
    n_components_list: List[int] | None = None,
    n_intervals: int = 10,
) -> List[Dict[str, Any]]:
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


def optimize_model_grid(
    X: np.ndarray,
    y: np.ndarray,
    wl: np.ndarray | None,
    classification: bool,
    methods: List[str],
    n_components_range: range,
    validation_method: str,
    validation_params: Dict[str, Any] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Dict[str, Any]:
    """Grid-search over preprocessing and number of components.

    Returns a dictionary with the list of ``results`` and the ``best`` entry.
    Each result contains top-level metrics (Accuracy/F1 for classification or
    RMSECV/R2 for regression) plus an ``id`` and ``rank`` field for the
    frontend.
    """

    validation_params = validation_params or {}
    splits = list(build_cv(validation_method, y, classification, validation_params))
    cv_splits = max(1, len(splits))
    print(
        f"[grid] mode={'classification' if classification else 'regression'}; validation={validation_method}; splits={len(splits)}",
        flush=True,
    )

    done = 0
    total_steps = 0
    grid_results: List[Dict[str, Any]] = []
    cache_Xp: dict[str, np.ndarray] = {}
    classes_global = np.unique(y) if classification else None

    for prep in (methods or ["none"]):
        if prep not in cache_Xp:
            Xp = apply_methods(X, methods=[prep] if prep != "none" else [], wl=wl)
            Xp, _ = sanitize_X(Xp)
            cache_Xp[prep] = Xp
        else:
            Xp = cache_Xp[prep]

        max_nc = int(min(Xp.shape[1], max(1, Xp.shape[0] - 1)))
        comp_range = [nc for nc in n_components_range if 1 <= nc <= max_nc]
        total_steps += len(comp_range) * cv_splits

        log_info(f"[grid] prep={prep} Xp.shape={Xp.shape} max_nc={max_nc} splits={len(splits)}")

        for nc in comp_range:
            all_true: list[Any] = []
            all_pred: list[Any] = []
            try:
                for tr, te in splits:
                    r = train_pls(
                        Xp[tr],
                        y[tr],
                        Xp[te],
                        y[te],
                        n_components=int(nc),
                        classification=bool(classification),
                        validation_method="none",
                        validation_params={},
                        all_labels=classes_global,
                    )
                    model = r.get("model")
                    if classification:
                        pred = model.predict(Xp[te])
                    else:
                        pred = model.predict(Xp[te]).ravel()
                    all_true.extend(y[te].tolist())
                    all_pred.extend(pred.tolist())

                if classification:
                    from .metrics import classification_metrics

                    m = classification_metrics(np.array(all_true), np.array(all_pred), labels=classes_global)
                    result = {
                        "id": f"{prep}__{nc}",
                        "prep": prep,
                        "n_components": int(nc),
                        "Accuracy": round(m.get("Accuracy", 0.0), 6),
                        "Kappa": round(m.get("Kappa", 0.0), 6),
                        "F1": round(m.get("F1", 0.0), 6),
                        "validation": {"method": validation_method},
                        "is_valid": True,
                    }
                else:
                    from .metrics import regression_metrics

                    m = regression_metrics(np.array(all_true), np.array(all_pred))
                    result = {
                        "id": f"{prep}__{nc}",
                        "prep": prep,
                        "n_components": int(nc),
                        "RMSECV": round(m.get("RMSE", 0.0), 6),
                        "R2": round(m.get("R2", 0.0), 6),
                        "validation": {"method": validation_method},
                        "is_valid": True,
                    }

                grid_results.append(result)

            except Exception as ex:
                log_info(f"[grid] skip prep={prep} n_comp={nc}: {type(ex).__name__}: {ex}")

            finally:
                done += cv_splits
                if progress_callback:
                    progress_callback(int(done), int(max(1, total_steps)))

    if classification:
        grid_sorted = sorted(
            grid_results,
            key=lambda r: (r.get("F1") or 0.0, r.get("Accuracy") or 0.0),
            reverse=True,
        )
    else:
        grid_sorted = sorted(grid_results, key=lambda r: r.get("RMSECV", float("inf")))

    for rank, r in enumerate(grid_sorted, start=1):
        r["rank"] = rank

    best = grid_sorted[0] if grid_sorted else None
    return {"results": grid_sorted, "best": best}

