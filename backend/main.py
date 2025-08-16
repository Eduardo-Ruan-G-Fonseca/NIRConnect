from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
import os, sys
import json
import pickle
import pandas as pd
import io
import numpy as np
import csv
from starlette.formparsers import MultiPartParser
from datetime import datetime
from pydantic import BaseModel, validator, Field
import logging
import warnings
import uuid
import time

from .logging_conf import *  # configure logging early

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn.metrics")

logger = logging.getLogger("nir")

if os.path.dirname(__file__) not in sys.path:
    sys.path.append(os.path.dirname(__file__))

from core.config import METRICS_FILE, settings
from core.metrics import regression_metrics, classification_metrics
from core.report_pdf import PDFReport
from core.logger import log_info
from collections import Counter
from core.validation import build_cv, make_cv, safe_n_components
from core.bootstrap import train_plsr, train_plsda, bootstrap_metrics
from core.preprocessing import apply_methods, sanitize_X, build_spectral_mask
from core.interpreter import interpretar_vips, gerar_resumo_interpretativo
from typing import Optional, Tuple, List, Literal, Any, Dict, Union
from core.saneamento import saneamento_global
from core.io_utils import to_float_matrix, encode_labels_if_needed
from core.optimization import optimize_model_grid, preprocess as grid_preprocess, make_pls_da, make_pls_reg
from ml.validation import build_cv as build_cv_meta
try:
    from ml.pipeline import (
        build_pls_pipeline,
        eval_pls_regression,
        eval_plsda_binary,
        eval_plsda_multiclass,
    )
except Exception:
    from core.ml.pipeline import (
        build_pls_pipeline,
        eval_pls_regression,
        eval_plsda_binary,
        eval_plsda_multiclass,
    )  # fallback se mover


from core.pls import is_categorical  # (se não for usar, podemos remover depois)
import joblib

# Progresso global para /optimize/status
OPTIMIZE_PROGRESS = {"current": 0, "total": 0}


class _State:
    """Simple container for keeping dataset and model between requests."""

    last_X: np.ndarray | None = None
    last_y: np.ndarray | None = None
    last_model: Any | None = None


state = _State()


def _metrics_ok(m):
    if m is None:
        return False
    if isinstance(m, dict):
        return any(np.isfinite(v) for v in m.values() if isinstance(v, (int, float)))
    return np.isfinite(m)


def apply_preprocess(X: np.ndarray, method: str) -> np.ndarray:
    if not method or method == "none":
        Xp = X.copy()
    else:
        Xp = apply_methods(X.copy(), [method])
    Xp, _ = sanitize_X(Xp)
    return Xp


def optimize_handler(X: np.ndarray, y: np.ndarray, params: dict, progress_callback=None):
    """Core optimization routine with error collection and fallback."""
    n_samples, n_features = X.shape
    task = "classification" if params.get("analysis_mode") == "PLS-DA" else "regression"

    cv = make_cv(
        method=params.get("validation_method"),
        params=params.get("validation_params"),
        n_samples=n_samples,
        task=task,
        y=y,
    )

    max_nc_req = int(params.get("n_components", 10))
    max_nc = safe_n_components(max_nc_req, n_samples=n_samples, n_features=n_features)

    try:
        n_splits = getattr(cv, "n_splits", None) or 1
    except Exception:
        n_splits = 1

    if task == "classification":
        _, counts = np.unique(y, return_counts=True)
        min_per_class = int(counts.min())
        log_info(f"[optimize] stratified=True splits={n_splits} min_per_class={min_per_class}")
    else:
        log_info(f"[optimize] splits={n_splits}")

    preps = params.get("preprocessing_methods") or ["none"]

    # Extra protection for heavy LOO combinations
    if params.get("validation_method") == "LeaveOneOut" and n_samples >= 80:
        preps = preps[:2]
        max_nc = min(max_nc, 10)

    analysis_mode = params.get("analysis_mode", "PLS-R")

    results = []
    errors = Counter()

    total_steps = len(preps) * max_nc
    done = 0

    for prep in preps:
        Xp = apply_preprocess(X, prep)
        for nc in range(1, max_nc + 1):
            try:
                if analysis_mode == "PLS-R":
                    m = eval_pls_regression(Xp, y, nc, cv)
                    if _metrics_ok(m):
                        results.append(
                            {
                                "preprocess": prep,
                                "n_components": nc,
                                "RMSECV": float(m["RMSECV"]),
                                "val_metrics": {"R2": float(m.get("R2", np.nan))},
                                "validation": {"method": params.get("validation_method")},
                                "wl_used": params.get("spectral_range") or [],
                            }
                        )
                    else:
                        errors["metric_invalid"] += 1
                else:  # PLS-DA
                    n_classes = len(np.unique(y))
                    if n_classes > 2:
                        m = eval_plsda_multiclass(Xp, y, nc, cv)
                        if _metrics_ok(m):
                            results.append(
                                {
                                    "preprocess": prep,
                                    "n_components": nc,
                                    "val_metrics": {
                                        "Accuracy": m["Accuracy"],
                                        "MacroF1": m["MacroF1"],
                                    },
                                    "confusion_matrix": m.get("confusion_matrix"),
                                    "labels": m.get("labels"),
                                    "validation": {"method": params.get("validation_method")},
                                    "wl_used": params.get("spectral_range") or [],
                                }
                            )
                        else:
                            errors["metric_invalid"] += 1
                    else:
                        m = eval_plsda_binary(Xp, y, nc, cv)
                        if _metrics_ok(m):
                            results.append(
                                {
                                    "preprocess": prep,
                                    "n_components": nc,
                                    "val_metrics": {"Accuracy": float(m["Accuracy"])},
                                    "confusion_matrix": m.get("confusion_matrix"),
                                    "labels": m.get("labels"),
                                    "validation": {"method": params.get("validation_method")},
                                    "wl_used": params.get("spectral_range") or [],
                                }
                            )
                        else:
                            errors["metric_invalid"] += 1
            except Exception as ex:  # noqa: B902
                errors[type(ex).__name__] += 1
            finally:
                done += 1
                if progress_callback:
                    progress_callback(done, total_steps)

    if not results:
        preps_fb = ["none"]
        nc_fb = min(10, max_nc)
        try:
            Xp = apply_preprocess(X, preps_fb[0])
            if analysis_mode == "PLS-R":
                m = eval_pls_regression(Xp, y, nc_fb, cv)
                if _metrics_ok(m):
                    results.append(
                        {
                            "preprocess": preps_fb[0],
                            "n_components": nc_fb,
                            "RMSECV": float(m["RMSECV"]),
                            "val_metrics": {"R2": float(m.get("R2", np.nan))},
                            "validation": {"method": params.get("validation_method")},
                            "wl_used": params.get("spectral_range") or [],
                        }
                    )
            else:
                n_classes = len(np.unique(y))
                if n_classes > 2:
                    m = eval_plsda_multiclass(Xp, y, nc_fb, cv)
                    if _metrics_ok(m):
                        results.append(
                            {
                                "preprocess": preps_fb[0],
                                "n_components": nc_fb,
                                "val_metrics": {
                                    "Accuracy": m["Accuracy"],
                                    "MacroF1": m["MacroF1"],
                                },
                                "confusion_matrix": m.get("confusion_matrix"),
                                "labels": m.get("labels"),
                                "validation": {"method": params.get("validation_method")},
                                "wl_used": params.get("spectral_range") or [],
                            }
                        )
                else:
                    m = eval_plsda_binary(Xp, y, nc_fb, cv)
                    if _metrics_ok(m):
                        results.append(
                            {
                                "preprocess": preps_fb[0],
                                "n_components": nc_fb,
                                "val_metrics": {"Accuracy": float(m["Accuracy"])},
                                "confusion_matrix": m.get("confusion_matrix"),
                                "labels": m.get("labels"),
                                "validation": {"method": params.get("validation_method")},
                                "wl_used": params.get("spectral_range") or [],
                            }
                        )
        except Exception as ex:  # noqa: B902
            errors[f"fallback_{type(ex).__name__}"] += 1

    return {"results": results, "errors_summary": dict(errors)}
app = FastAPI(title="NIR API v4.6")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    def scrub(x):
        if isinstance(x, (bytes, bytearray)):
            return f"<{len(x)} bytes>"
        if isinstance(x, dict):
            return {k: scrub(v) for k, v in x.items()}
        if isinstance(x, list):
            return [scrub(v) for v in x]
        return x
    return JSONResponse(
        status_code=422,
        content={
            "detail": scrub(exc.errors()),
            "hint": "Esta rota espera application/json. O upload de arquivo é apenas na etapa de importação.",
        },
    )


@app.on_event("startup")
async def _log_routes():
    for r in app.routes:
        methods = getattr(r, "methods", set())
        print("ROTA:", methods, r.path)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",   # Vite dev
        "http://localhost:4173", "http://127.0.0.1:4173",   # Vite preview
        "http://localhost:3000", "http://127.0.0.1:3000",   # React dev
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Aumenta limites de upload para 100 MB
MultiPartParser.spool_max_size = 100 * 1024 * 1024
MultiPartParser.max_part_size = 100 * 1024 * 1024

# Labels de métodos de pré-processamento
METHOD_LABELS = {
    "snv": "SNV",
    "msc": "MSC",
    "sg1": "1ª Derivada",
    "sg2": "2ª Derivada",
    "minmax": "Normalização Min-Max",
    "zscore": "Z-score",
    "ncl": "NCL",
    "vn": "Vector Norm",
}
ALL_PREPROCESS_METHODS = ["none"] + list(METHOD_LABELS.keys())

# Intervalos aproximados de bandas NIR para grupos químicos
CHEMICAL_RANGES = [
    (1200, 1300, "celulose"),
    (1300, 1500, "água"),
    (1500, 1700, "lignina"),
]

LOG_DIR = settings.logging_dir
HISTORY_FILE = os.path.join(settings.models_dir, "history.json")

class Metrics(BaseModel):
    R2: float
    RMSE: float
    Accuracy: float


MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "pls_pipeline.joblib")


class PreprocessRequest(BaseModel):
    X: List[List[float]]
    y: Optional[List[float]] = None
    features: Optional[List[str]] = None
    methods: Optional[List] = None


@app.post("/preprocess", tags=["Model"])
def preprocess(req: PreprocessRequest):
    X = np.asarray(req.X, dtype=float)
    nan_before = int(np.isnan(X).sum())
    if req.methods:
        from core.preprocessing import apply_methods
        X = apply_methods(X, req.methods)
    X_clean, y_clean, features = saneamento_global(X, req.y, req.features)
    nan_after = int(np.isnan(X_clean).sum())
    preview = X_clean[:5].tolist()
    return {
        "shape_before": list(np.asarray(req.X).shape),
        "shape_after": list(X_clean.shape),
        "nans_before": nan_before,
        "nans_after": nan_after,
        "preview": preview,
        "features": features,
        "y": y_clean.tolist() if y_clean is not None else None,
    }



class PredictRequest(BaseModel):
    X: List[List[Any]]


@app.post("/predict", tags=["Model"])
def predict(req: PredictRequest, threshold: float = 0.5):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=400, detail="Modelo não treinado")
    blob: Dict[str, Any] = joblib.load(MODEL_PATH)
    pipeline = blob["pipeline"]
    class_mapping = blob.get("class_mapping", {})
    X = to_float_matrix(req.X)
    scores = pipeline.predict(X).ravel()
    out: Dict[str, Any] = {"predictions": scores.tolist()}
    if class_mapping:
        idx = (scores >= threshold).astype(int).tolist()
        labels = [class_mapping.get(i, str(i)) for i in idx]
        out["labels_pred"] = labels
        out["class_mapping"] = class_mapping
    return out



def _latest_log() -> str:
    if not os.path.isdir(LOG_DIR):
        return ""
    logs = [f for f in os.listdir(LOG_DIR) if f.endswith(".log")]
    if not logs:
        return ""
    logs.sort(reverse=True)
    with open(os.path.join(LOG_DIR, logs[0]), "r") as f:
        return f.read()

import numpy as np
from sklearn.impute import SimpleImputer

def _fit_clean_X(X: np.ndarray):
    """
    Limpa X para treino:
      - força float
      - ±inf -> NaN
      - remove colunas 100% NaN e guarda o mask
      - imputa NaN por mediana (fit)
    Retorna: X_clean, col_mask, imputer
    """
    X = np.asarray(X, dtype=float)
    X[~np.isfinite(X)] = np.nan

    col_ok = ~np.isnan(X).all(axis=0)
    if not col_ok.any():
        raise HTTPException(status_code=400, detail="Todas as variáveis espectrais ficaram inválidas após o pré-processamento.")
    if not col_ok.all():
        X = X[:, col_ok]

    # linhas 100% NaN
    row_ok = ~np.isnan(X).all(axis=1)
    if not row_ok.any():
        raise HTTPException(status_code=400, detail="Todas as amostras ficaram inválidas após o pré-processamento.")
    if not row_ok.all():
        X = X[row_ok]

    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(X)
    return X, col_ok, imp, row_ok

def _transform_clean_X(X: np.ndarray, col_ok: np.ndarray, imp: SimpleImputer):
    """
    Limpa X de validação/teste usando a MESMA seleção de colunas e o MESMO imputador.
    """
    X = np.asarray(X, dtype=float)
    X[~np.isfinite(X)] = np.nan
    X = X[:, col_ok]
    # se alguma linha ficar toda NaN, o imputer ainda resolve (todas colunas NaN -> vira medianas)
    return imp.transform(X)

def _read_dataframe(filename: str, content: bytes) -> pd.DataFrame:
    """Read uploaded CSV or Excel into a DataFrame.

    This helper is tolerant to blank lines at the beginning of the file and to
    partially empty header rows. If the detected header only contains generic
    ``Unnamed`` columns, the first row is treated as data and the next row is
    used as header.
    """
    if filename.lower().endswith(".csv"):
        text = content.decode("utf-8", errors="ignore")
        try:
            delimiter = csv.Sniffer().sniff(text[:1024]).delimiter
        except Exception:
            delimiter = ","
        data = io.StringIO(text)
        df = pd.read_csv(data, delimiter=delimiter, header=0, skip_blank_lines=True)
        if all(str(c).startswith("Unnamed") or str(c).strip() == "" for c in df.columns):
            data.seek(0)
            df = pd.read_csv(data, delimiter=delimiter, header=None, skip_blank_lines=True)
            header = df.iloc[0].tolist()
            df = df.iloc[1:].reset_index(drop=True)
            df.columns = [str(h).strip() for h in header]
    else:
        bio = io.BytesIO(content)
        try:
            df = pd.read_excel(bio, sheet_name=0, header=0)
        except Exception:
            bio.seek(0)
            df = pd.read_excel(bio, sheet_name=0, engine="openpyxl", header=0)
        if all((str(c).startswith("Unnamed") or str(c).strip() == "" or pd.isna(c)) for c in df.columns):
            bio.seek(0)
            df = pd.read_excel(bio, sheet_name=0, header=None)
            header = df.iloc[0].tolist()
            df = df.iloc[1:].reset_index(drop=True)
            df.columns = [str(h).strip() for h in header]
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _to_numeric_series(s: pd.Series) -> pd.Series:
    s = s.astype(str)
    s = s.str.replace(r"\s+", "", regex=True)                 # remove espaços
    s = s.str.replace(r"[^0-9eE\+\-\,\.]", "", regex=True)    # mantém dígitos, sinais, . e ,
    s = s.str.replace(r"\.(?=\d{3}(?:\D|$))", "", regex=True) # remove ponto de milhar
    s = s.str.replace(",", ".", regex=False)                  # vírgula -> ponto
    return pd.to_numeric(s, errors="coerce")


def _cross_val_metrics(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int,
    classification: bool,
    n_splits: int = 5,
    validation_method: str | None = None,
    validation_params: dict | None = None,
    threshold: float = 0.5,
):
    from sklearn.impute import SimpleImputer

    # default de validação
    if validation_method is None:
        validation_method = "StratifiedKFold" if classification else "KFold"
        validation_params = {
            "n_splits": n_splits,
            "shuffle": True,
            "random_state": 42,
        }
    validation_params = validation_params or {}

    cv = build_cv(validation_method, y, classification, validation_params)

    # preds como objeto; converto no final
    preds = np.empty(len(y), dtype=object)
    y_series = pd.Series(y).astype(str)
    y_true = y_series.values

    for train_idx, test_idx in cv:
        # -----------------------
        # separa e saneia X do fold
        # -----------------------
        X_tr_raw = np.asarray(X[train_idx], dtype=float)
        X_te_raw = np.asarray(X[test_idx], dtype=float)

        # inf -> NaN
        X_tr_raw[~np.isfinite(X_tr_raw)] = np.nan
        X_te_raw[~np.isfinite(X_te_raw)] = np.nan

        # mantém só colunas que têm pelo menos um valor no TREINO
        col_ok = ~np.isnan(X_tr_raw).all(axis=0)
        if not col_ok.any():
            raise ValueError("Fold sem variáveis válidas após pré-processamento.")
        X_tr = X_tr_raw[:, col_ok]
        X_te = X_te_raw[:, col_ok]

        # remove linhas 100% NaN no TREINO (alinha y do treino)
        tr_row_ok = ~np.isnan(X_tr).all(axis=1)
        if not tr_row_ok.any():
            raise ValueError("Fold sem amostras válidas após pré-processamento.")
        X_tr = X_tr[tr_row_ok]

        # imputação por MEDIANA (fit no treino, aplica no teste)
        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X_tr)
        X_te = imp.transform(X_te)

        if classification:
            # y como string (classes)
            y_tr = pd.Series(y[train_idx]).astype(str).iloc[tr_row_ok].values
            model, _, extra = train_plsda(
                X_tr, y_tr, n_components=n_components
            )
            preds[test_idx] = model.predict(X_te)
        else:
            # y numérico no treino com o mesmo filtro de linhas
            y_tr_full = np.asarray(y[train_idx], dtype=float)
            y_tr = y_tr_full[tr_row_ok]
            model, _, _ = train_plsr(X_tr, y_tr, n_components=n_components)
            preds[test_idx] = model.predict(X_te).ravel()

    if classification:
        labels = sorted(pd.Series(y_true).unique())
        return classification_metrics(y_true, preds.astype(str), labels=labels)

    preds = preds.astype(float)
    y_float = np.asarray(y, dtype=float)
    return regression_metrics(y_float, preds)



def _parse_ranges(ranges: str, columns: list[str]) -> list[str]:
    """Return column names within specified wavelength ranges."""
    if not ranges:
        return columns
    selected: list[str] = []
    numeric_cols = []
    for c in columns:
        try:
            numeric_cols.append((float(c), c))
        except Exception:
            pass
    for part in ranges.split(','):
        part = part.strip()
        if not part or '-' not in part:
            continue
        start, end = part.split('-', 1)
        try:
            start_f = float(start)
            end_f = float(end)
        except ValueError:
            continue
        for val, name in numeric_cols:
            if start_f <= val <= end_f and name not in selected:
                selected.append(name)
    return selected if selected else columns


def _is_number(val: str) -> bool:
    """Return True if the value can be parsed as float."""
    try:
        float(val)
        return True
    except Exception:
        return False


def _parse_preprocess(preprocess: str | list | None) -> list[dict]:
    """Normalize preprocess parameter into a list of steps."""
    if not preprocess:
        return []
    if isinstance(preprocess, list):
        return preprocess
    try:
        data = json.loads(preprocess)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return [{"method": p.strip()} for p in str(preprocess).split(",") if p.strip()]


def _chemical_label(wl: float) -> str:
    for start, end, label in CHEMICAL_RANGES:
        if start <= wl <= end:
            return label
    return ""


def _append_history(entry: dict) -> None:
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    try:
        with open(HISTORY_FILE, 'r') as fh:
            data = json.load(fh)
    except Exception:
        data = []
    data.append(entry)
    with open(HISTORY_FILE, 'w') as fh:
        json.dump([jsonable_encoder(e) for e in data], fh, indent=2)


def _scatter_plot(y_real: list[float], y_pred: list[float]) -> str:
    import matplotlib.pyplot as plt
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    plt.figure()
    plt.scatter(y_real, y_pred)
    plt.xlabel("y_real")
    plt.ylabel("y_previsto")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

def _scores_plot(scores: list[list[float]], y: list) -> str:
    import matplotlib.pyplot as plt
    import numpy as np
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    arr = np.array(scores)
    if arr.shape[1] < 2:
        arr = np.column_stack([arr[:,0], np.zeros(arr.shape[0])])
    classes = np.unique(y)
    plt.figure()
    for cls in classes:
        idx = np.array(y) == cls
        plt.scatter(arr[idx,0], arr[idx,1], label=str(cls))
    plt.xlabel("Score 1")
    plt.ylabel("Score 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

def _cm_plot(y_true: list, y_pred: list, labels: list[str]) -> str:
    """Generate a confusion matrix image with clearer annotations."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import tempfile, os
    from sklearn.metrics import confusion_matrix

    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    n = len(labels)
    size = max(4, n * 0.6)
    fontsize = max(8, 14 - n)
    fig, ax = plt.subplots(figsize=(size, size), dpi=150)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Contagem"},
        square=True,
        annot_kws={"fontsize": fontsize},
        ax=ax,
    )
    ax.set_xlabel("Predito", fontsize=fontsize)
    ax.set_ylabel("Real", fontsize=fontsize)
    ax.set_title("Matriz de Confusão", fontsize=fontsize + 2, weight="bold")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

def _class_report_plot(report: dict) -> str:
    """Render classification report as an image."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import tempfile, os

    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    df = pd.DataFrame(report).transpose()
    fig, ax = plt.subplots(dpi=150)
    ax.axis("off")
    tbl = ax.table(
        cellText=df.round(3).values,
        colLabels=df.columns,
        rowLabels=df.index,
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    fig.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _vip_plot(vip: list[float]) -> str:
    import matplotlib.pyplot as plt
    import tempfile, os
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    plt.figure()
    plt.bar(range(len(vip)), vip)
    plt.xlabel("Variável")
    plt.ylabel("VIP")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/metrics/upload")
async def upload_metrics(metrics: Metrics) -> dict:
    try:
        os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
        with open(METRICS_FILE, "w", encoding="utf-8") as f:
            json.dump(metrics.dict(), f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "success", "message": "Métricas atualizadas"}


@app.post("/process")
async def process_file(
    file: UploadFile = File(...),
    target: str = Form(...),
    n_components: int = Form(5),
    classification: bool = Form(False),
    preprocess: str = Form(""),
    threshold: float = Form(0.5),
) -> dict:
    """Processa arquivo Excel/CSV para treinar um modelo PLS."""
    try:
        content = await file.read()
        df = _read_dataframe(file.filename, content)

        # 1) valida se a coluna alvo existe
        if target not in df.columns:
            raise HTTPException(status_code=400, detail=f"A coluna '{target}' não foi encontrada no arquivo.")

        # 2) prepara X/feature names uma vez só
        X_df = df.drop(columns=[target], errors="ignore")
        features = X_df.columns.tolist()
        X = X_df.values

        # 3) aplica pré-processamento, se houver
        methods = _parse_preprocess(preprocess)
        if methods:
            X = apply_methods(X, methods)

        X_tmp = np.asarray(X, dtype=float)
        X_tmp[~np.isfinite(X_tmp)] = np.nan
        row_ok = ~np.isnan(X_tmp).all(axis=1)
        X, col_ok = sanitize_X(X)
        features = [f for i, f in enumerate(features) if col_ok[i]]

        # 4) treina
        if classification:
            y_all = df[target].values  # classes como string/obj funcionam
            y = y_all[row_ok]
            _, metrics, extra = train_plsda(X, y, n_components=n_components)
        else:
            # garante numérico na regressão
            try:
                y_all = df[target].astype(float).values
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail="Erro: coluna alvo não numérica. Selecione uma coluna numérica para regressão.",
                )
            y = y_all[row_ok]
            _, metrics, extra = train_plsr(X, y, n_components=n_components)

        # 5) VIP + top Vips serializáveis
        vip = extra.get("vip", [])
        vip_list = vip.tolist() if hasattr(vip, "tolist") else list(vip)

        idx = np.argsort(vip_list)[::-1][:10]
        top_vips = []
        for i in idx:
            try:
                wl_float = float(features[i])
            except Exception:
                wl_float = float("nan")
            wl_value = wl_float if not np.isnan(wl_float) else features[i]
            label = _chemical_label(wl_float) if not np.isnan(wl_float) else ""
            top_vips.append({"wavelength": wl_value, "vip": float(vip_list[i]), "label": label})

        interpretacao = interpretar_vips(
            vip_list,
            [float(f) if str(f).replace('.', '', 1).isdigit() else None for f in features],
        )

        # resposta segura para JSON
        return jsonable_encoder({
            "metrics": metrics,
            "vip": vip_list,
            "top_vips": top_vips,
            "range_used": "",
            "interpretacao_vips": interpretacao
        })

    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - sanity
        raise HTTPException(status_code=400, detail=str(exc))

@app.post("/columns")
async def get_columns(file: UploadFile = File(...)) -> dict:
    """Return column metadata from uploaded file."""
    content = await file.read()
    try:
        df = _read_dataframe(file.filename, content)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail=("Erro ao ler as colunas da planilha. "
                    "Verifique se o arquivo contém um cabeçalho válido na primeira linha."),
        )

    if df.shape[1] == 0:
        raise HTTPException(
            status_code=400,
            detail=("Erro ao ler as colunas da planilha. "
                    "Verifique se o arquivo contém um cabeçalho válido na primeira linha."),
        )

    columns = [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns]

    # Detecta colunas espectrais (cabeçalho numérico)
    spectra: list[str] = []
    wls: list[float] = []
    warnings: list[str] = []
    for name in df.columns:
        s = str(name).strip()
        try:
            wl = float(s.replace(",", "."))
        except Exception:
            # tem dígito mas não é número puro -> avisar
            if any(ch.isdigit() for ch in s):
                warnings.append(f"Coluna '{s}' ignorada como espectro devido a formato inválido")
            continue
        spectra.append(s)
        wls.append(wl)

    # Alvo(s) possíveis: tudo que não é espectro
    targets = [c for c in df.columns if c not in spectra]

    mean_spec = {"wavelengths": [], "values": []}
    spectra_matrix = {"wavelengths": [], "values": []}

    if spectra:
        # DataFrame numérico com coerção
        numeric_df = df[spectra].apply(pd.to_numeric, errors="coerce")

        # 1) Ordena por comprimento de onda crescente e reordena colunas
        order = np.argsort(wls)
        wls_sorted = [float(wls[i]) for i in order]
        spectra_sorted = [spectra[i] for i in order]
        numeric_df = numeric_df[spectra_sorted]

        # 2) Média espectral
        means = numeric_df.mean().to_numpy()

        # 3) Substitui NaN por None (JSON-safe)
        numeric_df = numeric_df.where(pd.notnull(numeric_df), None)
        means = [None if (m is None or (isinstance(m, float) and np.isnan(m))) else float(m) for m in means]

        # 4) Preenche estruturas esperadas pelo front
        mean_spec["wavelengths"] = wls_sorted
        mean_spec["values"] = means
        spectra_matrix["wavelengths"] = wls_sorted
        spectra_matrix["values"] = numeric_df.values.tolist()

    # Retorno compatível com o front (usa targets / mean_spectra / spectra_matrix)
    return {
        "columns": columns,            # extra, útil para debug
        "targets": targets,            # usado no front
        "spectra": spectra,            # extra
        "mean_spectra": mean_spec,     # usado no front
        "spectra_matrix": spectra_matrix,  # usado no front
        "warnings": warnings,          # extra
    }
from starlette.concurrency import run_in_threadpool

@app.post("/analisar")
async def analisar_file(
    file: UploadFile = File(...),
    target: str = Form(...),
    n_components: int = Form(5),
    n_bootstrap: int = Form(0),
    classification: bool = Form(False),
    preprocess: str = Form(""),
    threshold: float = Form(0.5),
    decision_mode: str = Form("argmax"),
    spectral_ranges: str = Form("", alias="spectral_ranges"),
    validation_method: str | None = Form(None),
    validation_params: str = Form(""),
) -> dict:
    """Executa análise PLS básica retornando métricas e VIPs (robusto a vírgula decimal e NaN em X)."""
    try:
        payload = {
            "target": target,
            "n_components": n_components,
            "spectral_ranges": spectral_ranges,
            "preprocess": _parse_preprocess(preprocess),
            "n_bootstrap": n_bootstrap,
            "classification": classification,
            "threshold": threshold,
            "decision_mode": decision_mode,
            "validation_method": validation_method,
            "validation_params": validation_params,
        }
        log_info(f"Payload recebido: {payload}")

        try:
            val_params = json.loads(validation_params) if validation_params else {}
        except Exception:
            val_params = {}

        # =========================
        # Carrega dados
        # =========================
        content = await file.read()
        df = _read_dataframe(file.filename, content)
        if target not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"A coluna '{target}' não foi encontrada no arquivo."
            )

        # X / features (+ faixas)
        # =========================
        X_df = df.drop(columns=[target], errors="ignore")
        features = X_df.columns.tolist()
        X = to_float_matrix(X_df.values)

        wl_vals = []
        for c in features:
            try:
                wl_vals.append(float(str(c).strip().replace(",", ".")))
            except Exception:
                wl_vals.append(np.nan)
        wl_arr = np.array(wl_vals, dtype=float)

        ranges_list = []
        if spectral_ranges:
            for part in spectral_ranges.split(','):
                part = part.strip()
                if not part or '-' not in part:
                    continue
                start, end = part.split('-', 1)
                try:
                    ranges_list.append((float(start), float(end)))
                except Exception:
                    continue
        else:
            ranges_list = None

        mask = build_spectral_mask(wl_arr, ranges_list)
        X = X[:, mask]
        wl_arr = wl_arr[mask]
        features = [f for i, f in enumerate(features) if mask[i]]

        methods = _parse_preprocess(preprocess)
        if methods:
            X = apply_methods(X, methods)

        y_raw = df[target].tolist()
        if classification:
            y_arr, class_mapping, _ = encode_labels_if_needed(y_raw)
            X, y_arr, features = saneamento_global(X, y_arr, features)
            if class_mapping:
                y_series = pd.Series([class_mapping[int(i)] for i in y_arr.astype(int)])
            else:
                y_series = pd.Series(y_arr).astype(str)
        else:
            y_arr = pd.to_numeric(pd.Series(y_raw), errors="coerce").to_numpy(dtype=float)
            class_mapping = {}
            X, y_arr, features = saneamento_global(X, y_arr, features)
            y_series = pd.Series(y_arr)

        scores = None
        model = None
        metrics: dict = {}
        extra: dict = {}

        # --- HOLDOUT (tratamento especial)
        if validation_method == "Holdout":
            val_params = val_params or {"test_size": 0.2, "random_state": 42}
            cv = build_cv("Holdout", y_series.values, classification, val_params)
            train_idx, test_idx = next(cv)
            X_train_raw, X_test_raw = X[train_idx], X[test_idx]
            y_train_raw, y_test_raw = y_series.iloc[train_idx], y_series.iloc[test_idx]

            # LIMPEZA/IMPUTAÇÃO CONSISTENTE ENTRE TREINO/TESTE
            X_train, col_ok, imp, train_row_ok = _fit_clean_X(X_train_raw)
            # alinhar y de treino se houve remoção de linhas 100% NaN
            y_train = y_train_raw.iloc[train_row_ok].reset_index(drop=True)
            X_test = _transform_clean_X(X_test_raw, col_ok, imp)
            y_test = y_test_raw.reset_index(drop=True)

            if classification:
                model, train_metrics, extra = train_plsda(
                    X_train, y_train.values, n_components=n_components
                )
                y_test_pred = model.predict(X_test)
                classes = extra.get("classes", [])
                test_metrics = classification_metrics(y_test.astype(str).values, y_test_pred, labels=classes)

                metrics = {"train": train_metrics, "test": test_metrics}
                y_pred = y_test_pred.tolist()
                y_series_out = y_test.astype(str)
                X_out = X_test

            else:
                # regressão
                from sklearn.cross_decomposition import PLSRegression
                from sklearn.metrics import r2_score, mean_squared_error

                # alvo numérico robusto
                y_train_num = _to_numeric_series(y_train)
                y_test_num = _to_numeric_series(y_test)
                if y_train_num.isna().all() or y_test_num.isna().all():
                    raise HTTPException(status_code=400,
                                        detail="Erro: coluna alvo não numérica. Selecione uma coluna numérica para regressão.")

                # já está tudo limpo em X_train/X_test
                pls = PLSRegression(n_components=n_components)
                pls.fit(X_train, y_train_num.values)
                y_train_pred = pls.predict(X_train).ravel()
                y_test_pred = pls.predict(X_test).ravel()

                r2_cal = float(r2_score(y_train_num, y_train_pred))
                rmsec = float(np.sqrt(mean_squared_error(y_train_num, y_train_pred)))
                try:
                    r2_val = float(r2_score(y_test_num, y_test_pred))
                except Exception:
                    r2_val = None
                rmsep = float(np.sqrt(mean_squared_error(y_test_num, y_test_pred)))

                from core.metrics import vip_scores
                vip_list = vip_scores(pls, X_train, y_train_num.values.reshape(-1, 1)).tolist()
                extra = {"vip": vip_list, "scores": pls.x_scores_.tolist()}
                metrics = {"R2_cal": r2_cal, "RMSEC": rmsec, "R2_val": r2_val, "RMSEP": rmsep}
                model = pls

                y_pred = y_test_pred.tolist()
                y_series_out = y_test_num
                X_out = X_test
        # =========================
        # KFold / LOO / Sem validação explícita
        # =========================
        else:
            if classification:
                model, metrics, extra = await run_in_threadpool(
                    train_plsda, X, y_series.values,
                    n_components=n_components
                )
                y_pred = model.predict(X).tolist()
                y_series = y_series.astype(str)

                if validation_method in {"KFold", "LOO"}:
                    cvm = _cross_val_metrics(
                        X, y_series.values, n_components, classification=True,
                        validation_method=validation_method, validation_params=val_params
                    )
                    metrics = {"train": metrics, "cv": cvm}
            else:
                y_numeric = _to_numeric_series(y_series)
                if y_numeric.isna().all():
                    raise ValueError("Erro: coluna alvo não numérica. Por favor, selecione uma coluna com valores numéricos para regressão.")
                # alinha X ao y válido
                y_mask = y_numeric.notna().values
                X = X[y_mask]
                y_numeric = y_numeric[y_mask].values

                # saneamento-final em X (colunas+linhas)
                if not np.isfinite(X).all():
                    col_ok = np.isfinite(X).all(axis=0)
                    if not col_ok.any():
                        raise HTTPException(status_code=400, detail="Todas as variáveis ficaram inválidas após o pré-processamento.")
                    if not col_ok.all():
                        features = [f for i, f in enumerate(features) if col_ok[i]]
                        X = X[:, col_ok]
                    row_ok = np.isfinite(X).all(axis=1)
                    if not row_ok.any():
                        raise HTTPException(status_code=400, detail="Todas as amostras ficaram inválidas após o pré-processamento (NaN/Inf).")
                    X = X[row_ok]
                    y_numeric = y_numeric[row_ok]

                model, metrics, extra = await run_in_threadpool(
                    train_plsr, X, y_numeric, n_components=n_components
                )
                y_pred = model.predict(X).ravel().tolist()
                y_series = pd.Series(y_numeric)

                if validation_method in {"KFold", "LOO"}:
                    cvm = _cross_val_metrics(
                        X, y_numeric, n_components, classification=False,
                        validation_method=validation_method, validation_params=val_params
                    )
                    metrics = {"train": metrics, "cv": cvm}

        # =========================
        # Bootstrap opcional
        # =========================
        if n_bootstrap and int(n_bootstrap) > 0:
            boot = await run_in_threadpool(
                bootstrap_metrics, X, y_series.values,
                n_components=n_components, classification=classification, n_bootstrap=int(n_bootstrap)
            )
            metrics["bootstrap"] = boot

        # =========================
        # Salvar modelo e montar resposta
        # =========================
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        suffix = "plsda" if classification else "plsr"
        save_dir = settings.plsda_dir if classification else settings.plsr_dir
        os.makedirs(save_dir, exist_ok=True)
        model_name = f"modelo_{suffix}_{ts}.pkl"
        with open(os.path.join(save_dir, model_name), "wb") as fh:
            pickle.dump({"model": model, "class_mapping": class_mapping}, fh)

        vip = extra["vip"]
        features_for_vip = features  # já sincronizado quando removemos colunas
        idx = np.argsort(vip)[::-1][:10]
        top_vips = []
        wls_numeric = [float(f) if str(f).replace('.', '', 1).isdigit() else None for f in features_for_vip]
        for i in idx:
            try:
                wl_float = float(features_for_vip[i])
            except Exception:
                wl_float = None
            wl_value = wl_float if wl_float is not None else features_for_vip[i]
            label = _chemical_label(wl_float) if wl_float is not None else ""
            top_vips.append({"wavelength": wl_value, "vip": float(vip[i]), "label": label})

        interpretacao = interpretar_vips(vip, wls_numeric)
        resumo = gerar_resumo_interpretativo(interpretacao)
        method_names = [m["method"] if isinstance(m, dict) else m for m in methods]

        _append_history({
            "file": file.filename,
            "target": target,
            "tipo_analise": "PLS-DA" if classification else "PLS-R",
            "metrics": metrics,
            "preprocessing": [METHOD_LABELS.get(m, m) for m in method_names],
            "preprocess_steps": methods,
            "classes": np.unique(y_series).tolist() if classification else None,
            "class_mapping": class_mapping,
            "datetime": datetime.now().isoformat(timespec="seconds"),
            "top_vips": top_vips,
            "range_used": spectral_ranges if spectral_ranges else "",
            "model_name": model_name,
        })

        return jsonable_encoder({
            "metrics": metrics,
            "vip": vip,
            "y_real": y_series.tolist(),
            "y_pred": y_pred,
            "features": features_for_vip,
            "top_vips": top_vips,
            "range_used": spectral_ranges if spectral_ranges else "",
            "scores": extra.get("scores"),
            "analysis_type": "PLS-DA" if classification else "PLS-R",
            "model_name": model_name,
            "interpretacao_vips": interpretacao,
            "resumo_interpretativo": resumo,
            "class_mapping": class_mapping,
            "decision_mode": decision_mode,
        })

    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - sanity
        raise HTTPException(status_code=400, detail=str(exc))



# Alias de treino que usam o MESMO handler de /analisar (FormData)
# (removido na nova API)

class OptimizeParams(BaseModel):
    """Parameters expected by the /optimize endpoint."""
    target: str
    validation_method: Literal["KFold", "LOO", "Holdout"]
    n_components: Optional[int] = None          # se None, você define um padrão no /optimize
    n_bootstrap: int = 0
    folds: Optional[int] = None                 # usado só quando KFold
    analysis_mode: Literal["PLS-R", "PLS-DA"] = "PLS-R"
    spectral_range: Optional[Tuple[float, float]] = None
    preprocessing_methods: Optional[List[str]] = None

    @validator("n_components")
    def _check_ncomp(cls, v):
        if v is not None and v <= 0:
            raise ValueError("n_components deve ser > 0")
        return v

    @validator("folds")
    def _check_folds(cls, v, values):
        if values.get("validation_method") == "KFold":
            if v is None or v < 2:
                raise ValueError("Para KFold, 'folds' deve ser >= 2")
        return v

    @validator("spectral_range")
    def _order_range(cls, v):
        if v is None:
            return v
        a, b = v
        return (min(a, b), max(a, b))

@app.get("/optimize/status")
async def optimize_status() -> dict:
    """Return progress for current optimization."""
    return {
        "current": int(OPTIMIZE_PROGRESS.get("current", 0)),
        "total": int(OPTIMIZE_PROGRESS.get("total", 0)),
    }

import numpy as np

def _resolve_cv_params(raw_method, classification: bool, y_values: np.ndarray, requested_folds: int | None):
    """
    Resolve método e parâmetros de validação de forma segura/robusta.
    - LOO: retorna ("LOO", {})
    - StratifiedKFold (classificação): n_splits <= min(amostras por classe), min 2
    - KFold: n_splits <= n_samples, min 2
    - Holdout: usa defaults
    - Vazio/None: default = StratifiedKFold (classif) ou KFold (regr)
    """
    method = (raw_method or "").strip() if raw_method else None

    # LOO sempre respeitado
    if method == "LOO":
        return "LOO", {}

    # Defaults por tipo de análise
    if not method:
        method = "StratifiedKFold" if classification else "KFold"

    # Holdout
    if method == "Holdout":
        return "Holdout", {"test_size": 0.2, "shuffle": True, "random_state": 42}

    # StratifiedKFold (classificação)
    if method == "StratifiedKFold" and classification:
        folds = int(requested_folds) if requested_folds else 5
        _, counts = np.unique(y_values, return_counts=True)
        n_splits = int(min(max(2, folds), max(2, counts.min())))
        return "StratifiedKFold", {
            "n_splits": n_splits,
            "shuffle": True,
            "random_state": 42,
        }

    # KFold (regr ou class sem estratificação)
    if method == "KFold":
        folds = int(requested_folds) if requested_folds else 5
        n_samples = int(len(y_values))
        n_splits = int(min(max(2, folds), max(2, n_samples)))
        return "KFold", {"n_splits": n_splits, "shuffle": True, "random_state": 42}

    # fallback
    return method, {}

@app.post("/optimize-upload")
async def optimize_endpoint(
    file: UploadFile = File(...),
    params: str = Form("{}"),
) -> dict:
    """Run model optimization over preprocessing and PLS components."""
    try:
        # --- parse / valida params
        try:
            parsed = json.loads(params or "{}")
        except Exception:
            raise HTTPException(status_code=422, detail="Invalid parameters: params must be a JSON string")

        try:
            opts = OptimizeParams(**parsed)  # usa a versão com validators se você aplicou
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        content = await file.read()
        df = _read_dataframe(file.filename, content)
        if opts.target not in df.columns:
            raise HTTPException(status_code=400, detail="Target not found")

        # --- prepara X_df e filtra colunas espectrais numéricas
        X_df = df.drop(columns=[opts.target])

        numeric_cols: list[str] = []
        wls_vals: list[float] = []
        for c in X_df.columns:
            try:
                v = float(str(c).strip().replace(",", "."))
                numeric_cols.append(c)
                wls_vals.append(v)
            except Exception:
                pass
        if not numeric_cols:
            raise HTTPException(status_code=400, detail="Nenhuma coluna espectral (cabeçalho numérico) foi encontrada.")

        order = np.argsort(wls_vals)
        cols_sorted = [numeric_cols[i] for i in order]
        wls_sorted = [wls_vals[i] for i in order]

        X_df = X_df[cols_sorted]
        X = X_df.values
        wl = np.array(wls_sorted, dtype=float)

        sr = parsed.get("spectral_range")
        if sr:
            if isinstance(sr, dict):
                ranges = [(sr.get("start"), sr.get("end"))]
            elif isinstance(sr, (list, tuple)) and len(sr) >= 2:
                ranges = [(sr[0], sr[1])]
            else:
                ranges = None
            mask = build_spectral_mask(wl, ranges)
            X = X[:, mask]
            wl = wl[mask]

        # --- y e modo
        classification = opts.analysis_mode.upper() == "PLS-DA"
        y_series = df[opts.target]
        if classification:
            y = y_series.astype(str).values
            classes = np.unique(y)
            if len(classes) < 2:
                raise HTTPException(
                    status_code=400,
                    detail="A coluna alvo precisa ter pelo menos duas classes distintas para otimização."
                )
        else:
            try:
                y = y_series.astype(float).values
            except Exception:
                raise HTTPException(status_code=400, detail="Target must be numeric for regression (PLS-R).")

        # --- métodos de pré-processamento
        methods_in = opts.preprocessing_methods if opts.preprocessing_methods else ALL_PREPROCESS_METHODS
        methods = [m for m in methods_in if m in ALL_PREPROCESS_METHODS]
        if not methods:
            raise HTTPException(status_code=422, detail="Nenhum método de pré-processamento válido informado.")

        # --- componentes: define intervalo seguro conforme dimensões de X
        max_comp = int(min(X.shape[1], max(1, X.shape[0] - 1)))
        n_comp_range = list(range(1, max_comp + 1))
        log_info(f"[optimize] X shape={X.shape}, n_comp_range={n_comp_range}")

        # ===== Validação: grid leve + LOO final se solicitado =====
        raw_val_method = parsed.get("validation_method")
        requested_folds = parsed.get("folds")
        y_values = y_series.to_numpy()

        if raw_val_method == "LOO":
            grid_method, grid_params = _resolve_cv_params(
                raw_method=("StratifiedKFold" if classification else "KFold"),
                classification=bool(classification),
                y_values=y_values,
                requested_folds=(int(requested_folds) if requested_folds is not None else None),
            )
            final_method, final_params = "LOO", {}
        else:
            grid_method, grid_params = _resolve_cv_params(
                raw_method=raw_val_method,
                classification=bool(classification),
                y_values=y_values,
                requested_folds=(int(requested_folds) if requested_folds is not None else None),
            )
            final_method, final_params = grid_method, grid_params

        log_info(
            f"[optimize] grid_method={grid_method} grid_params={grid_params} final={final_method}"
        )

        # --- progresso
        try:
            cv_iter = list(build_cv(grid_method, y_values, bool(classification), grid_params))
            num_splits = int(max(1, len(cv_iter)))
        except Exception:
            num_splits = 1  # fallback

        methods_count = int(len(methods) if methods else 1)
        OPTIMIZE_PROGRESS["current"] = 0
        OPTIMIZE_PROGRESS["total"] = int(OPTIMIZE_PROGRESS.get("total") or 0)
        OPTIMIZE_PROGRESS["total"] = int(methods_count * len(n_comp_range) * num_splits)

        log_info(
            f"Otimizacao iniciada: cv={grid_method}, ncomp={max_comp}, preprocess={methods}"
        )

        try:
            opt_res = optimize_model_grid(
                X=X,
                y=y,
                wl=wl,
                classification=classification,
                methods=methods,
                n_components_range=n_comp_range,
                validation_method=grid_method,
                validation_params=grid_params,
                progress_callback=lambda c, t: OPTIMIZE_PROGRESS.update(
                    {"current": int(c), "total": int(t)}
                ),
            )
            results = opt_res.get("results", [])
            best = opt_res.get("best")
        finally:
            # garante finalização de progresso mesmo em erro
            OPTIMIZE_PROGRESS["current"] = int(OPTIMIZE_PROGRESS.get("total") or 0)

        if raw_val_method == "LOO" and results:
            best = best or results[0]
            best_prep = best.get("preprocess") or best.get("prep") or "none"
            best_nc = int(best.get("n_components") or best.get("components") or best.get("n_comp") or 1)

            from core.preprocessing import apply_methods, sanitize_X

            Xp = apply_methods(X, methods=[best_prep])
            Xp, _ = sanitize_X(Xp)

            from core.validation import build_cv
            splits = list(build_cv("LOO", y_values, bool(classification), {}))

            from core.pls import train_pls
            log_info(
                f"[optimize][final LOO] best_prep={best_prep}, best_nc={best_nc}, splits={len(splits)}"
            )

            scores: list[float] = []
            for tr, te in splits:
                r = train_pls(
                    Xp[tr],
                    y_values[tr],
                    Xp[te],
                    y_values[te],
                    n_components=best_nc,
                    classification=bool(classification),
                    validation_method="none",
                    validation_params={},
                )
                scores.append(
                    r["metrics"].get("F1") if classification else -r["metrics"].get("RMSE", np.nan)
                )
            results[0]["final_validation"] = {
                "method": "LOO",
                "score": float(np.mean(scores)) if scores else float("nan"),
            }

        return jsonable_encoder({"results": results[:15], "best": best})

    except HTTPException:
        raise
    except Exception as exc:
        # zera progresso em caso de erro inesperado
        OPTIMIZE_PROGRESS["current"] = 0
        OPTIMIZE_PROGRESS["total"] = 0
        raise HTTPException(status_code=400, detail=str(exc))

def generate_report(payload: dict) -> str:
    os.makedirs("reports", exist_ok=True)
    pdf = PDFReport()
    result = {
        "validation_used": payload.get("validation_used"),
        "n_splits_effective": payload.get("n_splits_effective"),
        "range_used": payload.get("range_used"),
        "best": payload.get("best"),
        "per_class": payload.get("per_class"),
        "curves": payload.get("curves"),
    }
    pdf.add_metrics(payload.get("metrics", {}), params=payload.get("params"), result=result)
    path = os.path.join("reports", f"relatorio_{uuid.uuid4().hex}.pdf")
    pdf.output(path)
    return path


@app.post("/report")
def report_endpoint(payload: dict):
    pdf_path = generate_report(payload)
    if not os.path.exists(pdf_path):
        raise HTTPException(500, "Falha ao gerar relatório.")
    return {"path": pdf_path}


@app.get("/report/download")
def report_download(path: str):
    if not os.path.exists(path):
        raise HTTPException(404, "Arquivo não encontrado.")
    return FileResponse(path, media_type="application/pdf", filename=os.path.basename(path))


@app.get("/metrics")
def get_metrics() -> Metrics:
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Metrics(**data)
    return Metrics(R2=0.0, RMSE=0.0, Accuracy=0.0)


@app.get("/dashboard/data")
async def dashboard_data(log_type: str = "", date: str = "") -> dict:
    logs = _latest_log() or ""
    filtered = [
        line for line in logs.splitlines()
        if (not log_type or log_type in line) and (not date or date in line)
    ]
    log_content = "\n".join(filtered)

    levels = ["INFO", "ERROR", "WARNING", "DEBUG"]
    counts = {lvl: log_content.count(lvl) for lvl in levels}

    # Usa a própria função da rota /metrics dentro do mesmo módulo
    metrics = get_metrics().dict()  # ok

    metric_history = {
        "dates": ["2025-07-20", "2025-07-21", "2025-07-22", "2025-07-23"],
        "r2":    [0.91, 0.93, 0.94, 0.95],
        "rmse":  [0.12, 0.11, 0.10, 0.09],
        "accuracy": [0.88, 0.89, 0.90, 0.91],
    }

    return {
        "logs": log_content[-5000:],   # limita tamanho da resposta
        "log_counts": counts,
        "model_metrics": metrics,
        "metric_history": metric_history,
    }


@app.get("/history/data")
async def history_data() -> list[dict]:
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as fh:
                return json.load(fh)
    except Exception:
        # Se arquivo estiver corrompido, retorna vazio em vez de 500
        return []
    return []


# ---------------------------------------------------------------------------
# New simplified training endpoints
# ---------------------------------------------------------------------------


class OptimizeRequest(BaseModel):
    target: str
    n_components: Optional[int] = None
    classification: bool = True
    threshold: float = 0.5
    n_bootstrap: int = 0
    validation_method: str = "StratifiedKFold"
    validation_params: Dict = Field(default_factory=dict)
    spectral_ranges: Optional[Union[str, List[Tuple[float, float]]]] = None


@app.post("/optimize")
def optimize(req: OptimizeRequest):
    try:
        X, y = state.last_X, state.last_y

        start_nm, end_nm = None, None
        if isinstance(req.spectral_ranges, str) and "-" in req.spectral_ranges:
            s, e = req.spectral_ranges.split("-")
            start_nm, end_nm = float(s), float(e)
        elif isinstance(req.spectral_ranges, list) and req.spectral_ranges:
            first = req.spectral_ranges[0]
            if isinstance(first, (list, tuple)) and len(first) == 2:
                start_nm, end_nm = float(first[0]), float(first[1])

        cv, val_name, n_splits = build_cv_meta(y, req.validation_method, req.validation_params)

        out = optimize_model_grid(
            X,
            y,
            mode="classification" if req.classification else "regression",
            preprocessors=None,
            n_components_max=req.n_components,
            validation_method=val_name,
            n_splits=n_splits,
            wavelength_range=(start_nm, end_nm) if start_nm is not None and end_nm is not None else None,
            logger=logger,
            time_budget_s=None,
        )

        best = out.get("best", {})
        best_prep = best.get("preprocess")
        best_nc = best.get("n_components")
        Xp = grid_preprocess(
            X,
            method=best_prep,
            range_nm=(start_nm, end_nm) if start_nm is not None and end_nm is not None else None,
        )
        if req.classification:
            best_estimator = make_pls_da(n_components=best_nc).fit(Xp, y)
        else:
            best_estimator = make_pls_reg(n_components=best_nc).fit(Xp, y)

        os.makedirs("models", exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        model_id = f"{stamp}_{uuid.uuid4().hex[:8]}"
        model_path = f"models/{model_id}.joblib"
        joblib.dump(best_estimator, model_path)
        with open(f"models/{model_id}.meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "target": req.target,
                    "classification": req.classification,
                    "preprocess": best_prep,
                    "n_components": best_nc,
                    "validation_used": val_name,
                    "range_used": out.get("range_used"),
                    "metrics": best.get("val_metrics"),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        return {
            "validation_used": val_name,
            "n_splits_effective": n_splits,
            "range_used": out.get("range_used"),
            "curves": out.get("curves"),
            "best": best,
            "per_class": best.get("val_metrics", {}).get("per_class")
            if req.classification
            else None,
            "model_id": model_id,
            "model_path": model_path,
        }
    except Exception as e:
        logger.exception("optimize failed")
        raise HTTPException(400, str(e))


@app.get("/model/download/{model_id}")
def model_download(model_id: str):
    path = f"models/{model_id}.joblib"
    if not os.path.exists(path):
        raise HTTPException(404, "Modelo não encontrado.")
    return FileResponse(path, media_type="application/octet-stream", filename=os.path.basename(path))


class TrainRequest(BaseModel):
    preprocess: Optional[str] = None
    n_components: int
    classification: bool = True
    threshold: float = 0.5
    validation_method: str = "StratifiedKFold"
    validation_params: Dict = Field(default_factory=dict)
    spectral_ranges: Optional[Union[str, List[Tuple[float, float]]]] = None


@app.post("/train")
def train(req: TrainRequest):
    try:
        X, y = state.last_X, state.last_y

        start_nm, end_nm = None, None
        if isinstance(req.spectral_ranges, str) and "-" in req.spectral_ranges:
            s, e = req.spectral_ranges.split("-")
            start_nm, end_nm = float(s), float(e)
        elif isinstance(req.spectral_ranges, list) and req.spectral_ranges:
            first = req.spectral_ranges[0]
            if isinstance(first, (list, tuple)) and len(first) == 2:
                start_nm, end_nm = float(first[0]), float(first[1])

        cv, val_name, n_splits = build_cv_meta(y, req.validation_method, req.validation_params)

        Xp = grid_preprocess(
            X,
            method=req.preprocess,
            range_nm=(start_nm, end_nm) if start_nm is not None and end_nm is not None else None,
        )
        if req.classification:
            est = make_pls_da(n_components=req.n_components).fit(Xp, y)
        else:
            est = make_pls_reg(n_components=req.n_components).fit(Xp, y)
        state.last_model = est

        return {
            "validation_used": val_name,
            "n_splits_effective": n_splits,
            "range_used": [start_nm, end_nm] if start_nm is not None and end_nm is not None else None,
        }
    except Exception as e:
        logger.exception("train failed")
        raise HTTPException(400, str(e))
