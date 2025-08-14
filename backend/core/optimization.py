from typing import List, Dict, Any, Callable
import numpy as np
import pandas as pd
from .validation import build_cv
from .pls import train_pls
from .preprocessing import apply_methods, sanitize_X
from .metrics import regression_metrics, classification_metrics, hotelling_t2
from .logger import log_info
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
    else:
        y = y.astype(float)

    if preprocess_opts is None:
        preprocess_opts = ["none"]
    results: List[Dict[str, Any]] = []
    if validation_method is None:
        validation_method = "StratifiedKFold" if classification else "KFold"
        validation_params = {
            "n_splits": n_splits,
            "shuffle": True,
            "random_state": 42,
        }
    validation_params = validation_params or {}
    total_steps = len(preprocess_opts) * len(n_components_range)
    log_info(f"Otimizacao: {total_steps} combinacoes possiveis")
    done = 0
    for prep in preprocess_opts:

        if prep == "none" or prep == "":
            Xp = X.copy()
        else:
            try:
                Xp = apply_methods(X.copy(), [prep])
            except Exception as e:
                log_info(f"[grid] failed preprocess {prep}: {e}")
                done += len(n_components_range)
                if progress_callback:
                    progress_callback(done, total_steps)
                continue

        X_tmp = np.asarray(Xp, dtype=float)
        X_tmp[~np.isfinite(X_tmp)] = np.nan
        row_ok = ~np.isnan(X_tmp).all(axis=1)
        if not row_ok.any():
            log_info(f"[grid] skip prep={prep}: all samples invalid")
            done += len(n_components_range)
            if progress_callback:
                progress_callback(done, total_steps)
            continue
        Xp = X_tmp[row_ok]
        y_p = y[row_ok]
        try:
            if wl is not None:
                Xp, wl_list = sanitize_X(Xp, wl.tolist())
                wl_used = np.array(wl_list, dtype=float)
            else:
                Xp, _ = sanitize_X(Xp)
                wl_used = wl
            log_info(f"[grid] after sanitize: Xp.shape={Xp.shape}, nan={np.isnan(Xp).any()}")
        except ValueError as e:
            log_info(f"[grid] skip prep={prep}: {e}")
            done += len(n_components_range)
            if progress_callback:
                progress_callback(done, total_steps)
            continue
        if np.nanvar(Xp) < 1e-12 or Xp.shape[0] < 2 or Xp.shape[1] < 1:
            log_info(f"[grid] skip prep={prep}: degenerate matrix")
            done += len(n_components_range)
            if progress_callback:
                progress_callback(done, total_steps)
            continue
        max_nc = int(min(Xp.shape[1], max(1, Xp.shape[0] - 1)))
        comp_range = [nc for nc in n_components_range if 1 <= nc <= max_nc]
        if not comp_range:
            log_info(f"[grid] skip prep={prep}: no viable components (max_nc={max_nc})")
            done += len(n_components_range)
            if progress_callback:
                progress_callback(done, total_steps)
            continue
        for nc in comp_range:
            log_info(f"[grid] trying prep={prep}, n_comp={nc}, Xp={Xp.shape}")
            try:
                model, train_metrics, extra = train_pls(
                    Xp,
                    y_p,
                    n_components=nc,
                    classification=classification,
                    validation_method="none",
                )
                if validation_method == "LOO":
                    cv = build_cv("LOO", y_p, classification, {})
                else:
                    cv = build_cv(validation_method, y_p, classification, validation_params)
                preds = np.empty(len(y_p), dtype=float if not classification else object)
                for tr, te in cv:
                    m, _, _ = train_pls(
                        Xp[tr],
                        y_p[tr],
                        n_components=nc,
                        classification=classification,
                        validation_method="none",
                    )
                    pr = m.predict(Xp[te])
                    preds[te] = np.array(pr).ravel()
                if classification:
                    y_series = pd.Series(y_p).astype(str)
                    labels = sorted(y_series.unique())
                    val_metrics = classification_metrics(
                        y_series.values, preds.astype(str), labels=labels
                    )
                    rmsecv = None
                else:
                    preds_f = preds.astype(float)
                    val_metrics = regression_metrics(y_p, preds_f)
                    rmsecv = float(np.sqrt(np.mean((y_p - preds_f) ** 2)))
                T = np.array(extra.get("scores", []))
                leverage = (
                    np.diag(T @ np.linalg.pinv(T.T @ T) @ T.T).tolist() if T.size else []
                )
                ht2 = hotelling_t2(model.model).tolist()
                mean_t = T.mean(axis=0) if T.size else 0
                inv_cov = np.linalg.pinv(np.cov(T, rowvar=False)) if T.size else np.array([])
                mahal = (
                    [float((ti - mean_t) @ inv_cov @ (ti - mean_t).T) for ti in T]
                    if T.size
                    else []
                )
                results.append(
                    {
                        "preprocess": prep,
                        "n_components": nc,
                        "train_metrics": train_metrics,
                        "val_metrics": val_metrics,
                        "RMSECV": rmsecv,
                        "leverage": leverage,
                        "hotelling_t2": ht2,
                        "mahalanobis": mahal,
                        "wl_used": wl_used.tolist() if wl_used is not None else None,
                        "validation": {
                            "method": validation_method,
                            "params": validation_params,
                        },
                    }
                )
                log_info(
                    f"[Optimize] RMSECV: {rmsecv if rmsecv is not None else 'NA'}, R2: {val_metrics.get('R2', 'NA')}"
                )
            except Exception as e:
                log_info(f"[grid] failed prep={prep}, n_comp={nc}: {e}")
                continue
            finally:
                done += 1
                if progress_callback:
                    progress_callback(done, total_steps)
        if len(comp_range) < len(n_components_range):
            done += len(n_components_range) - len(comp_range)
            if progress_callback:
                progress_callback(done, total_steps)
    key = (
        lambda r: r["RMSECV"]
        if r["RMSECV"] is not None
        else -r["val_metrics"].get("Accuracy", 0)
    )
    log_info(f"[Optimize] Sucesso: {len(results)} combinacoes testadas")
    return sorted(results, key=key)
