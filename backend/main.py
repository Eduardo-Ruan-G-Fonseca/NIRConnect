from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form, Body
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.encploders import jsonable_encoder
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import json
import pickle
import pandas as pd
import io
import numpy as np
from starlette.formparsers import MultiPartParser
from datetime import datetime

from core.config import METRICS_FILE, settings
from core.metrics import regression_metrics, classification_metrics
from core.report_pdf import PDFReport
from core.logger import log_info
from core.validation import build_cv
from core.bootstrap import train_plsr, train_plsda, bootstrap_metrics
from core.preprocessing import apply_methods
from core.optimization import optimize_model_grid
from core.interpreter import interpretar_vips, gerar_resumo_interpretativo
from core.pls import is_categorical

app = FastAPI(title="NIR API v4.6")

# Monta templates e arquivos estáticos
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

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

LOG_DIR = settings.logging_dir
HISTORY_FILE = os.path.join(settings.models_dir, "history.json")


class Metrics(BaseModel):
    R2: float
    RMSE: float
    Accuracy: float


def _latest_log() -> str:
    if not os.path.isdir(LOG_DIR):
        return ""
    logs = [f for f in os.listdir(LOG_DIR) if f.endswith(".log")]
    if not logs:
        return ""
    logs.sort(reverse=True)
    with open(os.path.join(LOG_DIR, logs[0]), "r") as f:
        return f.read()


def _read_dataframe(filename: str, content: bytes) -> pd.DataFrame:
    """Read uploaded CSV or Excel into a DataFrame.

    This helper is tolerant to blank lines at the beginning of the file and to
    partially empty header rows. If the detected header only contains generic
    ``Unnamed`` columns, the first row is treated as data and the next row is
    used as header.
    """
    if filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content), header=0, skip_blank_lines=True)
        if all(str(c).startswith("Unnamed") or str(c).strip() == "" for c in df.columns):
            df = pd.read_csv(io.BytesIO(content), header=None, skip_blank_lines=True)
            header = df.iloc[0].tolist()
            df = df.iloc[1:].reset_index(drop=True)
            df.columns = [str(h).strip() for h in header]
    else:
        df = pd.read_excel(io.BytesIO(content), sheet_name=0, header=0)
        if all((str(c).startswith("Unnamed") or str(c).strip() == "" or pd.isna(c)) for c in df.columns):
            df = pd.read_excel(io.BytesIO(content), sheet_name=0, header=None)
            header = df.iloc[0].tolist()
            df = df.iloc[1:].reset_index(drop=True)
            df.columns = [str(h).strip() for h in header]
    df.columns = [str(c).strip() for c in df.columns]
    return df


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
    if validation_method is None:
        validation_method = "StratifiedKFold" if classification else "KFold"
        validation_params = {
            "n_splits": n_splits,
            "shuffle": True,
            "random_state": 42,
        }
    validation_params = validation_params or {}
    cv = build_cv(validation_method, y, classification, validation_params)
    preds = np.empty(len(y), dtype=object)
    y_series = pd.Series(y).astype(str)
    y_true = y_series.values
    for train_idx, test_idx in cv:
        if classification:
            model, _, extra = train_plsda(
                X[train_idx],
                y[train_idx],
                n_components=n_components,
                threshold=threshold,
            )
            Yp = model.predict(X[test_idx])
            if Yp.ndim > 1 and Yp.shape[1] > 1:
                idx = np.argmax(Yp, axis=1)
            else:
                idx = (Yp.ravel() > threshold).astype(int)
            classes = extra.get("classes", [])
            preds[test_idx] = np.array([classes[i] for i in idx])
        else:
            model, _, _ = train_plsr(
                X[train_idx], y[train_idx], n_components=n_components
            )
            preds[test_idx] = model.predict(X[test_idx]).ravel()
    if classification:
        labels = sorted(pd.Series(y_true).unique())
        return classification_metrics(y_true, preds.astype(str), labels=labels)
    preds = preds.astype(float)
    return regression_metrics(y, preds.astype(float))


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
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))
        X = df.drop(columns=[target]).values
        methods = _parse_preprocess(preprocess)
        if methods:
            X = apply_methods(X, methods)
        y = df[target].values
        if classification:
            _, metrics, extra = train_plsda(
                X, y, n_components=n_components, threshold=threshold
            )
        else:
            _, metrics, extra = train_plsr(X, y, n_components=n_components)
        vip = extra["vip"]
        features = df.drop(columns=[target]).columns.tolist()
        idx = np.argsort(vip)[::-1][:10]
        top_vips = []
        for i in idx:
            try:
                wl_float = float(features[i])
            except Exception:
                wl_float = float("nan")
            wl_value = wl_float if not np.isnan(wl_float) else features[i]
            label = _chemical_label(wl_float) if not np.isnan(wl_float) else ""
            top_vips.append({"wavelength": wl_value, "vip": float(vip[i]), "label": label})
        interpretacao = interpretar_vips(
            vip,
            [float(f) if str(f).replace('.', '', 1).isdigit() else None for f in features],
        )
        return {"metrics": metrics, "vip": vip, "top_vips": top_vips, "range_used": "", "interpretacao_vips": interpretacao}
    except HTTPException:
        raise
    except HTTPException:
        raise
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
            detail="Erro ao ler as colunas da planilha. Verifique se o arquivo contém um cabeçalho válido na primeira linha.",
        )
    if len(df.columns) == 0:
        raise HTTPException(
            status_code=400,
            detail="Erro ao ler as colunas da planilha. Verifique se o arquivo contém um cabeçalho válido na primeira linha.",
        )
    columns = [{"name": c, "dtype": str(df[c].dtype)} for c in df.columns]

    spectra: list[str] = []
    wls: list[float] = []
    warnings: list[str] = []
    for name in df.columns:
        s = str(name).strip()
        try:
            wl = float(s.replace(",", "."))
        except Exception:
            if any(ch.isdigit() for ch in s):
                warnings.append(
                    f"Coluna '{s}' ignorada como espectro devido a formato inválido"
                )
            continue
        spectra.append(s)
        wls.append(wl)

    targets = [c for c in df.columns if c not in spectra]

    mean_spec = {"wavelengths": [], "values": []}
    spectra_matrix = {"wavelengths": [], "values": []}
    if spectra:
        numeric_df = df[spectra].apply(pd.to_numeric, errors="coerce")
        means = numeric_df.mean().to_dict()
        for col, wl in zip(spectra, wls):
            mean_spec["wavelengths"].append(wl)
            mean_spec["values"].append(float(means.get(col, 0.0)))
        spectra_matrix["wavelengths"] = wls
        spectra_matrix["values"] = numeric_df.values.tolist()
    return {
        "columns": columns,
        "targets": targets,
        "spectra": spectra,
        "mean_spectra": mean_spec,
        "spectra_matrix": spectra_matrix,
        "warnings": warnings,
    }


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
    """Executa análise PLS básica retornando métricas e VIPs."""
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
        print("Payload recebido:", payload)
        try:
            val_params = json.loads(validation_params) if validation_params else {}
        except Exception:
            val_params = {}
        required = [
            "target",
            "n_components",
            "spectral_ranges",
            "n_bootstrap",
            "classification",
            "decision_mode",
        ]
        for key in required:
            val = payload.get(key)
            if val is None or (isinstance(val, (str, list)) and not val):
                raise HTTPException(status_code=422, detail=f"Campo '{key}' ausente ou inválido")

        content = await file.read()
        df = _read_dataframe(file.filename, content)
        if target not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"A coluna '{target}' não foi encontrada no arquivo."
            )
        X_df = df.drop(columns=[target])
        if spectral_ranges:
            cols = _parse_ranges(spectral_ranges, X_df.columns.tolist())
            X_df = X_df[cols]
        X = X_df.values
        methods = _parse_preprocess(preprocess)
        if methods:
            X = apply_methods(X, methods)
        y_series = df[target]

        if validation_method == "Holdout":
            val_params = val_params or {"test_size": 0.2, "random_state": 42}
            cv = build_cv("Holdout", y_series.values, classification, val_params)
            train_idx, test_idx = next(cv)
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_series.iloc[train_idx], y_series.iloc[test_idx]
            if classification:
                model, train_metrics, extra = train_plsda(
                    X_train, y_train.values, n_components=n_components, threshold=threshold
                )
                Yp_test = model.predict(X_test)
                if Yp_test.ndim > 1 and Yp_test.shape[1] > 1:
                    idx_test = np.argmax(Yp_test, axis=1)
                else:
                    idx_test = (Yp_test.ravel() > threshold).astype(int)
                classes = extra.get("classes", [])
                y_test_pred = np.array([classes[i] for i in idx_test])
                test_metrics = classification_metrics(y_test.astype(str).values, y_test_pred, labels=classes)
                metrics = {"train": train_metrics, "test": test_metrics}
                y_pred = y_test_pred.tolist()
                y_series = y_test.astype(str)
                X = X_test
            else:
                try:
                    y_train_num = y_train.astype(float).values
                    y_test_num = y_test.astype(float).values
                except Exception:
                    raise ValueError(
                        "Erro: coluna alvo não numérica. Por favor, selecione uma coluna com valores numéricos para regressão."
                    )
                from sklearn.cross_decomposition import PLSRegression
                from sklearn.metrics import r2_score, mean_squared_error
                pls = PLSRegression(n_components=n_components)
                pls.fit(X_train, y_train_num)
                y_train_pred = pls.predict(X_train).ravel()
                y_test_pred = pls.predict(X_test).ravel()
                r2_cal = r2_score(y_train_num, y_train_pred)
                rmsec = np.sqrt(mean_squared_error(y_train_num, y_train_pred))
                try:
                    r2_val = r2_score(y_test_num, y_test_pred)
                except Exception:
                    r2_val = float("nan")
                rmsep = np.sqrt(mean_squared_error(y_test_num, y_test_pred))
                from core.metrics import vip_scores
                vip = vip_scores(pls, X_train, y_train_num.reshape(-1,1)).tolist()
                scores = pls.x_scores_.tolist()
                extra = {"vip": vip, "scores": scores}
                metrics = {
                    "R2_cal": float(r2_cal),
                    "RMSEC": float(rmsec),
                    "R2_val": float(r2_val),
                    "RMSEP": float(rmsep),
                }
                model = pls
                y_pred = y_test_pred.tolist()
                y_series = pd.Series(y_test_num)
                X = X_test
        else:
            if classification:
                model, metrics, extra = train_plsda(
                    X, y_series.values, n_components=n_components, threshold=threshold
                )
                Y_pred = model.predict(X)
                if Y_pred.ndim > 1 and Y_pred.shape[1] > 1:
                    idx = np.argmax(Y_pred, axis=1)
                else:
                    idx = (Y_pred.ravel() > threshold).astype(int)
                classes = extra.get("classes", [])
                y_pred = [classes[i] for i in idx]
                y_series = y_series.astype(str)
            else:
                try:
                    y_numeric = y_series.astype(float).values
                except Exception:
                    raise ValueError(
                        "Erro: coluna alvo não numérica. Por favor, selecione uma coluna com valores numéricos para regressão."
                    )
                model, metrics, extra = train_plsr(X, y_numeric, n_components=n_components)
                y_pred = model.predict(X).ravel().tolist()
                y_series = pd.Series(y_numeric)
        if n_bootstrap and int(n_bootstrap) > 0:
            boot = bootstrap_metrics(
                X,
                y_series.values,
                n_components=n_components,
                classification=classification,
                n_bootstrap=int(n_bootstrap),
            )
            metrics["bootstrap"] = boot
        # Salvar modelo treinado
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        suffix = "plsda" if classification else "plsr"
        save_dir = settings.plsda_dir if classification else settings.plsr_dir
        os.makedirs(save_dir, exist_ok=True)
        model_name = f"modelo_{suffix}_{ts}.pkl"
        with open(os.path.join(save_dir, model_name), "wb") as fh:
            pickle.dump(model, fh)
        vip = extra["vip"]
        scores = extra.get("scores")
        features = X_df.columns.tolist()
        idx = np.argsort(vip)[::-1][:10]
        top_vips = []
        wls_numeric = [float(f) if str(f).replace('.', '', 1).isdigit() else None for f in features]
        for i in idx:
            try:
                wl_float = float(features[i])
            except Exception:
                wl_float = None
            wl_value = wl_float if wl_float is not None else features[i]
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
            "class_mapping": {int(i): cls for i, cls in enumerate(extra.get("classes", []))} if classification else None,
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
            "features": features,
            "top_vips": top_vips,
            "range_used": spectral_ranges if spectral_ranges else "",
            "scores": scores,
            "analysis_type": "PLS-DA" if classification else "PLS-R",
            "model_name": model_name,
            "interpretacao_vips": interpretacao,
            "resumo_interpretativo": resumo,
            "class_mapping": {int(i): cls for i, cls in enumerate(extra.get("classes", []))} if classification else None,
            "decision_mode": decision_mode,
        })
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - sanity
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/analyze")
async def analyze_file(
    file: UploadFile = File(...),
    target: str = Form(...),
    n_components: int = Form(5),
    classification: bool = Form(False),
    n_bootstrap: int = Form(0),
    preprocess: str = Form(""),
    threshold: float = Form(0.5),
) -> dict:
    """Execute PLS analysis with cross-validation and optional bootstrap."""
    try:
        content = await file.read()
        df = _read_dataframe(file.filename, content)
        X_df = df.drop(columns=[target])
        X = X_df.values
        methods = _parse_preprocess(preprocess)
        if methods:
            X = apply_methods(X, methods)
        y = df[target].values
        if classification:
            model, train_metrics, extra = train_plsda(
                X, y, n_components=n_components, threshold=threshold
            )
        else:
            model, train_metrics, extra = train_plsr(X, y.astype(float), n_components=n_components)
        cv_metrics = _cross_val_metrics(
            X,
            y,
            n_components,
            classification,
            threshold=threshold,
            validation_method=validation_method,
            validation_params=val_params,
        )
        metrics = {"train": train_metrics, "cv": cv_metrics}
        if n_bootstrap and int(n_bootstrap) > 0:
            boot = bootstrap_metrics(X, y, n_components=n_components, classification=classification, n_bootstrap=int(n_bootstrap))
            metrics["bootstrap"] = boot
        wls_numeric = [float(f) if str(f).replace('.', '', 1).isdigit() else None for f in X_df.columns.tolist()]
        interpretacao = interpretar_vips(extra["vip"], wls_numeric)
        return {"metrics": metrics, "vip": extra["vip"], "features": X_df.columns.tolist(), "interpretacao_vips": interpretacao}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


class OptimizeParams(BaseModel):
    """Parameters expected by the /optimize endpoint."""

    target: str
    validation_method: str
    n_components: int | None = None
    n_bootstrap: int = 0
    folds: int | None = None
    analysis_mode: str = "PLS-R"
    spectral_range: tuple[float, float] | None = None
    preprocessing_methods: list[str] | None = None


@app.get("/optimize/status")
async def optimize_status() -> dict:
    """Return progress for current optimization."""
    return OPTIMIZE_PROGRESS


@app.post("/optimize")
async def optimize_endpoint(
    file: UploadFile = File(...),
    params: str = Form("{}"),
) -> dict:
    """Run model optimization over preprocessing and PLS components."""
    try:
        try:
            parsed = json.loads(params or "{}")
        except Exception:
            raise HTTPException(status_code=422, detail="Invalid parameters")
        try:
            opts = OptimizeParams(**parsed)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        content = await file.read()
        df = _read_dataframe(file.filename, content)
        if opts.target not in df.columns:
            raise HTTPException(status_code=400, detail="Target not found")
        X_df = df.drop(columns=[opts.target])
        if opts.spectral_range:
            start, end = opts.spectral_range
            cols = [c for c in X_df.columns if _is_number(c) and start <= float(c) <= end]
            X_df = X_df[cols]
        X = X_df.values
        wl = np.array([float(c) for c in X_df.columns.astype(str)])
        classification = opts.analysis_mode.upper() == "PLS-DA"
        if classification:
            y_series = df[opts.target]
            y = y_series.astype(str).values
            log_info(f"y convertido em texto com shape {y.shape}")
            classes = np.unique(y)
            if len(classes) < 2:
                raise HTTPException(
                    status_code=400,
                    detail="A coluna alvo precisa ter pelo menos duas classes distintas para otimização.",
                )
        else:
            y = df[opts.target].values.astype(float)
        methods = (
            opts.preprocessing_methods if opts.preprocessing_methods else ALL_PREPROCESS_METHODS
        )
        max_comp = opts.n_components or min(X.shape[1], 10)
        OPTIMIZE_PROGRESS["current"] = 0
        OPTIMIZE_PROGRESS["total"] = len(methods) * max_comp
        cv_method = opts.validation_method or "KFold"
        val_params = {}
        if cv_method == "KFold":
            val_params = {
                "n_splits": opts.folds or 5,
                "shuffle": True,
                "random_state": 42,
            }
        log_info(
            f"Otimizacao iniciada: cv={cv_method}, ncomp={max_comp}, preprocess={methods}"
        )
        results = optimize_model_grid(
            X,
            y,
            wl,
            classification=classification,
            preprocess_opts=methods,
            n_components_range=range(1, max_comp + 1),
            validation_method="LOO" if cv_method == "LOO" else "KFold",
            validation_params=val_params,
            progress_callback=lambda c, t: OPTIMIZE_PROGRESS.update({"current": c, "total": t}),
        )
        return {"results": results[:15]}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/report")
async def create_report(data: dict = Body(...)):
    """Generate PDF report for given metrics and parameters."""
    metrics = data.get("metrics", {})
    params = data.get("params", {})
    y_real = data.get("y_real")
    y_pred = data.get("y_pred")
    vip = data.get("vip")
    top_vips = data.get("top_vips")
    interpretacao_vips = data.get("interpretacao_vips")
    resumo_interpretativo = data.get("resumo_interpretativo")
    scores = data.get("scores")
    analysis_type = params.get("analysis_type", "PLS-R")
    class_report_path = ""
    if analysis_type == "PLS-DA" and scores and y_real is not None and y_pred is not None:
        scatter = _scores_plot(scores, y_real)
        labels = []
        if isinstance(params.get("class_mapping"), dict):
            labels = [v for k, v in sorted(params["class_mapping"].items(), key=lambda x: int(x[0]))]
        else:
            labels = list(np.unique(y_real))
        conf_path = _cm_plot(y_real, y_pred, labels) if y_real and y_pred else ""
        if metrics.get("ClassificationReport"):
            class_report_path = _class_report_plot(metrics["ClassificationReport"])
    else:
        scatter = _scatter_plot(y_real, y_pred) if y_real and y_pred else ""
        conf_path = ""
    vip_path = _vip_plot(vip) if vip else ""
    pdf = PDFReport()
    pdf.add_metrics(
        metrics,
        params=params,
        scatter_path=scatter,
        vip_path=vip_path,
        conf_path=conf_path,
        class_report_path=class_report_path,
        user=params.get("user", ""),
        top_vips=top_vips,
        range_used=params.get("range_used", ""),
        interpretacao_vips=interpretacao_vips,
        resumo_interpretativo=resumo_interpretativo,
    )
    pdf_bytes = pdf.pdf.output(dest="S").encode("latin1")
    return StreamingResponse(io.BytesIO(pdf_bytes), media_type="application/pdf", headers={"Content-Disposition": "attachment; filename=report.pdf"})


@app.get("/metrics")
def get_metrics() -> Metrics:
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            data = json.load(f)
        return Metrics(**data)
    return Metrics(R2=0.0, RMSE=0.0, Accuracy=0.0)


@app.get("/dashboard/data")
async def dashboard_data(log_type: str = "", date: str = "") -> dict:
    logs = _latest_log()
    filtered = [
        line
        for line in logs.splitlines()
        if (not log_type or log_type in line) and (not date or date in line)
    ]
    log_content = "\n".join(filtered)

    levels = ["INFO", "ERROR", "WARNING", "DEBUG"]
    counts = {lvl: log_content.count(lvl) for lvl in levels}

    metrics = get_metrics().dict()

    metric_history = {
        "dates": ["2025-07-20", "2025-07-21", "2025-07-22", "2025-07-23"],
        "r2": [0.91, 0.93, 0.94, 0.95],
        "rmse": [0.12, 0.11, 0.10, 0.09],
        "accuracy": [0.88, 0.89, 0.90, 0.91],
    }

    return {
        "logs": log_content[-5000:],
        "log_counts": counts,
        "model_metrics": metrics,
        "metric_history": metric_history,
    }


@app.get("/history/data")
async def history_data() -> list[dict]:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as fh:
            return json.load(fh)
    return []

# Rotas de interface web
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("nir_interface.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request):
    return templates.TemplateResponse("history.html", {"request": request})


@app.get("/nir", response_class=HTMLResponse)
async def nir_interface(request: Request):
    return templates.TemplateResponse("nir_interface.html", {"request": request})
