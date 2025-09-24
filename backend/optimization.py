from __future__ import annotations

from itertools import product
from typing import Any, Dict, Iterable, List

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from preprocessing import apply_preprocessing
from utils.task_detect import detect_task_from_y
from utils.sanitize import sanitize_y
from validation import build_cv
from pls import sanitize_pls_inputs, cap_n_components


def _as_list(value: Iterable[Any] | Any) -> List[Any]:
    if value is None:
        return [None]
    if isinstance(value, (list, tuple, set)):
        return list(value) or [None]
    if isinstance(value, str):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _iter_grid(grid: Dict[str, Iterable[Any]]) -> Iterable[Dict[str, Any]]:
    if not grid:
        yield {}
        return

    control_keys = {"metric", "selection_metric"}
    keys = [k for k in grid.keys() if k not in control_keys]
    if not keys:
        yield {}
        return

    value_lists = [_as_list(grid[k]) for k in keys]
    for combo in product(*value_lists):
        yield {k: combo[i] for i, k in enumerate(keys)}


def _resolve_preprocess_ops(params: Dict[str, Any]) -> Dict[str, Any]:
    ops = {}
    value = (
        params.get("preprocess")
        or params.get("preprocess_steps")
        or params.get("preprocess_ops")
        or params.get("ops")
        or params.get("pipeline")
    )

    def _mark(step):
        if isinstance(step, str):
            key = step.strip().lower()
            if key == "snv":
                ops["SNV"] = True
            elif key == "msc":
                ops["MSC"] = True
            elif key.startswith("sg1"):
                ops.setdefault("SG1", {})
            elif key.startswith("sg0") or key.startswith("sg"):
                ops.setdefault("SG0", {})
        elif isinstance(step, dict):
            for k, v in step.items():
                if k.upper() in {"SNV", "MSC", "SG0", "SG1"}:
                    ops[k.upper()] = v

    if isinstance(value, dict):
        for k, v in value.items():
            if k.upper() in {"SNV", "MSC", "SG0", "SG1"}:
                ops[k.upper()] = v
    elif isinstance(value, (list, tuple)):
        for step in value:
            _mark(step)
    else:
        _mark(value)

    sg_params = params.get("sg") or params.get("sg_params")
    if sg_params is not None:
        try:
            window, poly, deriv = sg_params
            target = "SG1" if int(deriv or 0) else "SG0"
            ops[target] = {"window": int(window), "poly": int(poly)}
        except Exception:
            pass

    return ops


def _init_output(task: str, selection_metric: str) -> Dict[str, Any]:
    return {
        "status": "error",
        "message": "sem avaliação",
        "history": [],
        "task": task,
        "selection_metric": selection_metric,
    }


def optimize_model_grid(
    X: np.ndarray,
    y: np.ndarray,
    grid: Dict[str, Iterable[Any]],
    mode: str,
    validation_method: str = "KFold",
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    selection_metric = str(grid.get("selection_metric") or grid.get("metric") or "balanced_accuracy")
    metric_key = selection_metric.lower()

    X_clean, y_masked, base_row_mask, _ = sanitize_pls_inputs(X, y)
    if y_masked is not None:
        y_aligned = y_masked
    else:
        y_arr = np.asarray(y)
        y_aligned = y_arr[base_row_mask]

    task = detect_task_from_y(y_aligned, mode)
    y_enc, classes_ = sanitize_y(y_aligned, task)

    if X_clean.shape[0] == 0:
        return {"status": "error", "message": "Sem amostras após sanitização."}

    stratified = task == "classification"

    metric_lower = metric_key
    maximize = True
    if task != "classification":
        maximize = False
    else:
        if any(bad in metric_lower for bad in ("rmse", "mse", "mae", "loss", "erro")):
            maximize = False

    out = _init_output(task, selection_metric)

    best_score = -np.inf if maximize else np.inf
    best_payload: Dict[str, Any] | None = None

    combos = list(_iter_grid(grid))
    if not combos:
        default_nc = _as_list(grid.get("n_components", [2]))[0]
        combos = [{"n_components": default_nc, "preprocess": None}]

    for params in combos:
        ops = _resolve_preprocess_ops(params)
        try:
            X_proc = apply_preprocessing(X_clean, ops or {})
            X_proc = np.asarray(X_proc, dtype=float)
        except Exception as exc:
            out["history"].append({"params": params, "error": str(exc)})
            continue

        X_proc, _, combo_row_mask, _ = sanitize_pls_inputs(X_proc, None)
        if combo_row_mask.size and combo_row_mask.shape[0] == y_enc.shape[0]:
            y_proc = y_enc[combo_row_mask]
        else:
            y_proc = y_enc

        if X_proc.shape[0] < 2 or X_proc.shape[1] == 0:
            out["history"].append({"params": params, "error": "Dados insuficientes após pré-processamento."})
            continue

        unique_labels = np.unique(y_proc)
        if task == "classification" and unique_labels.size < 2:
            out["history"].append({"params": params, "error": "Apenas uma classe presente."})
            continue

        n_requested = int(params.get("n_components") or params.get("components") or params.get("k") or 2)
        if n_requested < 1:
            n_requested = 1

        try:
            oof_pred = np.full(y_proc.shape[0], np.nan)
            oof_proba = None
            if task == "classification":
                n_classes = int(unique_labels.size)
                oof_proba = np.zeros((y_proc.shape[0], n_classes), dtype=float)

            fold_scores: List[float] = []
            fold_balanced: List[float] = []
            used_components: List[int] = []

            cv_local = build_cv(
                validation_method,
                y=y_proc,
                n_splits=n_splits,
                stratified=stratified,
                random_state=random_state,
            )

            for train_idx, test_idx in cv_local.split(X_proc, y_proc):
                Xtr, Xte = X_proc[train_idx], X_proc[test_idx]
                ytr, yte = y_proc[train_idx], y_proc[test_idx]

                imputer = SimpleImputer(strategy="median")
                scaler = StandardScaler()
                Xtr_imp = imputer.fit_transform(Xtr)
                Xte_imp = imputer.transform(Xte)

                Xtr_scaled = scaler.fit_transform(Xtr_imp)
                Xte_scaled = scaler.transform(Xte_imp)

                safe_n = cap_n_components(n_requested, Xtr_scaled)
                if safe_n < 1:
                    raise ValueError("n_components inválido após sanitização.")
                used_components.append(safe_n)

                if task == "classification":
                    n_classes = oof_proba.shape[1]
                    Ytr = np.eye(n_classes)[ytr]
                    pls = PLSRegression(n_components=safe_n)
                    pls.fit(Xtr_scaled, Ytr)
                    preds = pls.predict(Xte_scaled)
                    if preds.ndim == 1:
                        preds = preds.reshape(-1, 1)
                    if preds.shape[1] == 1 and n_classes == 2:
                        prob_pos = np.clip(preds.ravel(), 0.0, 1.0)
                        proba = np.column_stack([1.0 - prob_pos, prob_pos])
                    else:
                        proba = np.clip(preds, 0.0, 1.0)
                        row_sum = proba.sum(axis=1, keepdims=True)
                        valid = row_sum.squeeze() > 0
                        if np.any(valid):
                            proba[valid] = proba[valid] / row_sum[valid]
                        if np.any(~valid):
                            proba[~valid] = 1.0 / n_classes

                    yhat = np.argmax(proba, axis=1)
                    oof_pred[test_idx] = yhat
                    oof_proba[test_idx] = proba
                    acc = accuracy_score(yte, yhat)
                    bal = balanced_accuracy_score(yte, yhat)
                    fold_scores.append(acc)
                    fold_balanced.append(bal)
                else:
                    pls = PLSRegression(n_components=safe_n)
                    pls.fit(Xtr_scaled, ytr)
                    pred = pls.predict(Xte_scaled).ravel()
                    oof_pred[test_idx] = pred
                    rmse = float(np.sqrt(mean_squared_error(yte, pred)))
                    fold_scores.append(rmse)

        except Exception as exc:
            out["history"].append({"params": params, "error": str(exc)})
            continue

        metrics = {}
        if task == "classification":
            metrics["accuracy"] = float(np.mean(fold_scores)) if fold_scores else float("nan")
            metrics["balanced_accuracy"] = float(np.mean(fold_balanced)) if fold_balanced else float("nan")
            current_score = metrics.get(metric_key)
            if current_score is None:
                current_score = metrics["balanced_accuracy"]
        else:
            metrics["rmse"] = float(np.mean(fold_scores)) if fold_scores else float("nan")
            current_score = metrics["rmse"]

        history_entry = {
            "params": params,
            "metrics": metrics,
            "used_n_components": int(np.median(used_components)) if used_components else None,
        }
        out["history"].append(history_entry)

        if not np.isfinite(current_score):
            continue

        is_better = current_score > best_score if maximize else current_score < best_score

        if is_better or best_payload is None:
            best_score = current_score
            best_payload = {
                "params": params,
                "metrics": metrics,
                "oof_pred": oof_pred.tolist() if oof_pred is not None else None,
                "proba_oof": oof_proba.tolist() if oof_proba is not None else None,
                "used_n_components": history_entry["used_n_components"],
            }

    if best_payload:
        out.update(
            {
                "status": "ok",
                "best_params": best_payload["params"],
                "best_score": best_score,
                "metric": selection_metric,
                "classes_": classes_ or [],
                "oof_pred": best_payload.get("oof_pred"),
                "proba_oof": best_payload.get("proba_oof"),
            }
        )
    else:
        out["message"] = "Nenhuma combinação válida avaliada."

    return out

