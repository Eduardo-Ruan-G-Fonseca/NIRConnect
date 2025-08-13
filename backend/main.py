from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body, APIRouter
from fastapi.responses import StreamingResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import pickle
import pandas as pd
import io
import numpy as np
import csv
from starlette.formparsers import MultiPartParser
from datetime import datetime
from pydantic import BaseModel, validator, Field

from core.config import METRICS_FILE, settings
from core.metrics import regression_metrics, classification_metrics
from core.report_pdf import PDFReport
from core.logger import log_info
from core.validation import build_cv
from core.bootstrap import train_plsr, train_plsda, bootstrap_metrics
from core.preprocessing import apply_methods
from core.optimization import optimize_model_grid
from core.interpreter import interpretar_vips, gerar_resumo_interpretativo
from typing import Optional, Tuple, List, Literal
from utils.saneamento import saneamento_global
from ml.pipeline import build_pls_pipeline


from core.pls import is_categorical  # (se não for usar, podemos remover depois)
import joblib

# Progresso global para /optimize/status
OPTIMIZE_PROGRESS = {"current": 0, "total": 0}

app = FastAPI(title="NIR API v4.6")
model_router = APIRouter(tags=["Model"])


app.include_router(model_router)

app.include_router(model_router.router)


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


class TrainRequest(BaseModel):
    X: List[List[float]]
    y: List[float]
    features: Optional[List[str]] = None
    n_components: int = Field(10, ge=1)
    n_splits: int = Field(5, ge=2)


@app.post("/train", tags=["Model"])

def train(req: TrainRequest):
    X_clean, y_clean, features = saneamento_global(req.X, req.y, req.features)
    if not np.isfinite(X_clean).all():
        raise HTTPException(status_code=400, detail="Dados contêm valores não finitos após saneamento")
    if np.isnan(X_clean).sum() != 0:
        raise HTTPException(status_code=400, detail="Dados contêm NaN após saneamento")
    if X_clean.shape[1] == 0:
        raise HTTPException(status_code=400, detail="Nenhuma coluna válida para treino")

    pipeline = build_pls_pipeline(req.n_components)
    from sklearn.model_selection import KFold, cross_validate

    cv = KFold(n_splits=req.n_splits, shuffle=True, random_state=42)
    cv_results = cross_validate(
        pipeline,
        X_clean,
        y_clean,
        cv=cv,
        scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error"},
        return_train_score=False,
    )
    r2_scores = cv_results["test_r2"].tolist()
    rmse_scores = (-cv_results["test_rmse"]).tolist()

    pipeline.fit(X_clean, y_clean)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"pipeline": pipeline, "features": features}, MODEL_PATH)

    return {
        "r2": r2_scores,
        "rmse": rmse_scores,
        "r2_mean": float(np.mean(r2_scores)),
        "rmse_mean": float(np.mean(rmse_scores)),
        "features": features,
    }


class PredictRequest(BaseModel):
    X: List[List[float]]



@app.post("/predict", tags=["Model"])

def predict(req: PredictRequest):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=400, detail="Modelo não treinado")
    model_data = joblib.load(MODEL_PATH)
    pipeline = model_data["pipeline"]
    X = np.asarray(req.X, dtype=float)
    preds = pipeline.predict(X).ravel().tolist()
    return {"predictions": preds}



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
                X_tr, y_tr, n_components=n_components, threshold=threshold
            )
            Yp = model.predict(X_te)
            if Yp.ndim > 1 and Yp.shape[1] > 1:
                idx = np.argmax(Yp, axis=1)
            else:
                idx = (Yp.ravel() > threshold).astype(int)
            classes = extra.get("classes", [])
            preds[test_idx] = np.array([classes[i] for i in idx])
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
        X_df = df.drop(columns=[target])
        features = X_df.columns.tolist()
        X = X_df.values

        # 3) aplica pré-processamento, se houver
        methods = _parse_preprocess(preprocess)
        if methods:
            X = apply_methods(X, methods)

        # 4) treina
        if classification:
            y = df[target].values  # classes como string/obj funcionam
            _, metrics, extra = train_plsda(X, y, n_components=n_components, threshold=threshold)
        else:
            # garante numérico na regressão
            try:
                y = df[target].astype(float).values
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail="Erro: coluna alvo não numérica. Selecione uma coluna numérica para regressão."
                )
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

        # =========================
        # X / features (+ faixas)
        # =========================
        X_df = df.drop(columns=[target])
        if spectral_ranges:
            cols = _parse_ranges(spectral_ranges, X_df.columns.tolist())
            X_df = X_df[cols]

        # força numérico nas features (qualquer sujeira vira NaN)
        X_df = X_df.apply(pd.to_numeric, errors="coerce")
        features = X_df.columns.tolist()
        X = X_df.values

        # pré-processamento (pode introduzir NaN dependendo do método)
        methods = _parse_preprocess(preprocess)
        if methods:
            X = apply_methods(X, methods)

        # =========================
        # Saneamento pós-preprocess (resolve NaN/Inf em X)
        #  - remove COLUNAS 100% NaN
        #  - remove LINHAS 100% NaN
        #  - imputa NaNs restantes por MEDIANA
        #  - alinha y às linhas mantidas
        # =========================
        import numpy as np
        from sklearn.impute import SimpleImputer

        # garante array float e troca ±inf por NaN
        X = np.asarray(X, dtype=float)
        X[~np.isfinite(X)] = np.nan

        # 1) drop de colunas 100% NaN (não têm mediana)
        col_ok = ~np.isnan(X).all(axis=0)
        if not col_ok.any():
            raise HTTPException(
                status_code=400,
                detail="Todas as variáveis espectrais ficaram inválidas após o pré-processamento."
            )
        if not col_ok.all():
            X = X[:, col_ok]
            features = [f for i, f in enumerate(features) if col_ok[i]]

        # 2) drop de linhas 100% NaN
        row_ok = ~np.isnan(X).all(axis=1)
        if not row_ok.any():
            raise HTTPException(
                status_code=400,
                detail="Todas as amostras ficaram inválidas após o pré-processamento."
            )
        if not row_ok.all():
            X = X[row_ok]
            row_mask_for_y = row_ok
        else:
            row_mask_for_y = None

        # 3) imputação de NaNs restantes com mediana da coluna
        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(X)  # agora X está 100% finito (sem NaN/Inf)

        # 4) alinhar y às linhas mantidas
        if row_mask_for_y is not None:
            y_series = df[target].iloc[row_mask_for_y].reset_index(drop=True)
        else:
            y_series = df[target].reset_index(drop=True)

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
                    n_components=n_components, threshold=threshold
                )
                Y_pred = model.predict(X)
                if Y_pred.ndim > 1 and Y_pred.shape[1] > 1:
                    idx = np.argmax(Y_pred, axis=1)
                else:
                    idx = (Y_pred.ravel() > threshold).astype(int)
                classes = extra.get("classes", [])
                y_pred = [classes[i] for i in idx]
                y_series = y_series.astype(str)

                if validation_method in {"KFold", "LOO"}:
                    cvm = _cross_val_metrics(
                        X, y_series.values, n_components, classification=True,
                        validation_method=validation_method, validation_params=val_params, threshold=threshold
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
            pickle.dump(model, fh)

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
            "features": features_for_vip,
            "top_vips": top_vips,
            "range_used": spectral_ranges if spectral_ranges else "",
            "scores": extra.get("scores"),
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


# ✅ Mantém para testes internos (opcional). O front usa /analisar.
@app.post("/analyze")
async def analyze_file(
    file: UploadFile = File(...),
    target: str = Form(...),
    n_components: int = Form(5),
    classification: bool = Form(False),
    n_bootstrap: int = Form(0),
    preprocess: str = Form(""),
    threshold: float = Form(0.5),
    # ⬇ adicionados para CV
    validation_method: str | None = Form(None),          # "KFold" | "LOO" | None
    validation_params: str = Form(""),                   # ex.: {"n_splits":5,"shuffle":true}
) -> dict:
    """Execute PLS analysis with cross-validation and optional bootstrap."""
    try:
        content = await file.read()
        df = _read_dataframe(file.filename, content)

        if target not in df.columns:
            raise HTTPException(status_code=400, detail=f"A coluna '{target}' não foi encontrada no arquivo.")

        X_df = df.drop(columns=[target])
        X = X_df.values

        methods = _parse_preprocess(preprocess)
        if methods:
            X = apply_methods(X, methods)

        # y e treino
        if classification:
            y = df[target].astype(str).values
            model, train_metrics, extra = train_plsda(X, y, n_components=n_components, threshold=threshold)
        else:
            try:
                y = df[target].astype(float).values
            except Exception:
                raise HTTPException(status_code=400, detail="Erro: coluna alvo não numérica para regressão.")
            model, train_metrics, extra = train_plsr(X, y, n_components=n_components)

        # CV opcional (KFold/LOO)
        try:
            val_params = json.loads(validation_params) if validation_params else {}
        except Exception:
            val_params = {}
        cv_metrics = None
        if validation_method in {"KFold", "LOO"}:
            cv_metrics = _cross_val_metrics(
                X, y, n_components, classification,
                validation_method=validation_method, validation_params=val_params,
                threshold=threshold
            )

        metrics = {"train": train_metrics}
        if cv_metrics is not None:
            metrics["cv"] = cv_metrics

        # bootstrap opcional
        if n_bootstrap and int(n_bootstrap) > 0:
            boot = bootstrap_metrics(X, y, n_components=n_components, classification=classification, n_bootstrap=int(n_bootstrap))
            metrics["bootstrap"] = boot

        # VIP/interpretacao
        vip_raw = extra.get("vip", [])
        vip_list = vip_raw.tolist() if hasattr(vip_raw, "tolist") else list(vip_raw)
        wls_numeric = [float(f) if str(f).replace('.', '', 1).isdigit() else None for f in X_df.columns.tolist()]
        interpretacao = interpretar_vips(vip_list, wls_numeric)

        return jsonable_encoder({
            "metrics": metrics,
            "vip": vip_list,
            "features": X_df.columns.tolist(),
            "interpretacao_vips": interpretacao
        })
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

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

@app.post("/optimize")
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

        # helper: detecta colunas com cabeçalho numérico
        numeric_cols = []
        for c in X_df.columns:
            try:
                float(str(c).strip().replace(",", "."))
                numeric_cols.append(c)
            except Exception:
                pass
        if not numeric_cols:
            raise HTTPException(status_code=400, detail="Nenhuma coluna espectral (cabeçalho numérico) foi encontrada.")

        # aplica spectral_range se houver
        if opts.spectral_range:
            start, end = opts.spectral_range
            filtered = []
            for c in numeric_cols:
                v = float(str(c).strip().replace(",", "."))
                if start <= v <= end:
                    filtered.append(c)
            if not filtered:
                raise HTTPException(status_code=400, detail="Nenhuma coluna dentro do intervalo espectral informado.")
            numeric_cols = filtered

        # reordena por comprimento de onda crescente
        wls = [float(str(c).strip().replace(",", ".")) for c in numeric_cols]
        order = np.argsort(wls)
        cols_sorted = [numeric_cols[i] for i in order]
        wls_sorted = [wls[i] for i in order]

        X_df = X_df[cols_sorted]
        X = X_df.values
        wl = np.array(wls_sorted, dtype=float)

        # --- y e modo
        classification = opts.analysis_mode.upper() == "PLS-DA"
        if classification:
            y_series = df[opts.target]
            y = y_series.astype(str).values
            classes = np.unique(y)
            if len(classes) < 2:
                raise HTTPException(
                    status_code=400,
                    detail="A coluna alvo precisa ter pelo menos duas classes distintas para otimização."
                )
        else:
            try:
                y = df[opts.target].astype(float).values
            except Exception:
                raise HTTPException(status_code=400, detail="Target must be numeric for regression (PLS-R).")

        # --- métodos de pré-processamento
        methods_in = opts.preprocessing_methods if opts.preprocessing_methods else ALL_PREPROCESS_METHODS
        methods = [m for m in methods_in if m in ALL_PREPROCESS_METHODS]
        if not methods:
            raise HTTPException(status_code=422, detail="Nenhum método de pré-processamento válido informado.")

        # --- componentes (cap por n_features)
        max_comp = opts.n_components or min(X.shape[1], 10)
        max_comp = max(1, min(max_comp, X.shape[1]))

        # --- validação: só KFold/LOO na otimização
        cv_method = (opts.validation_method or "KFold").upper()
        if cv_method == "HOLDOUT":
            raise HTTPException(status_code=422, detail="Holdout não é suportado na otimização. Use KFold ou LOO.")
        val_params = {}
        if cv_method == "KFOLD":
            val_params = {
                "n_splits": opts.folds or 5,
                "shuffle": True,
                "random_state": 42,
            }

        # --- progresso
        OPTIMIZE_PROGRESS["current"] = 0
        OPTIMIZE_PROGRESS["total"] = len(methods) * max_comp

        log_info(f"Otimizacao iniciada: cv={cv_method}, ncomp={max_comp}, preprocess={methods}")

        try:
            results = optimize_model_grid(
                X,
                y,
                wl,
                classification=classification,
                preprocess_opts=methods,
                n_components_range=range(1, int(max_comp) + 1),
                validation_method="LOO" if cv_method == "LOO" else "KFold",
                validation_params=val_params,
                progress_callback=lambda c, t: OPTIMIZE_PROGRESS.update({"current": int(c), "total": int(t)}),
            )
        finally:
            # garante finalização de progresso mesmo em erro
            OPTIMIZE_PROGRESS["current"] = OPTIMIZE_PROGRESS.get("total", 0)

        # corta para 15 e garante JSON-safe
        return jsonable_encoder({"results": results[:15]})

    except HTTPException:
        raise
    except Exception as exc:
        # zera progresso em caso de erro inesperado
        OPTIMIZE_PROGRESS["current"] = 0
        OPTIMIZE_PROGRESS["total"] = 0
        raise HTTPException(status_code=400, detail=str(exc))

@app.post("/report")
async def create_report(data: dict = Body(...)):
    """Generate PDF report for given metrics and parameters."""
    try:
        metrics = data.get("metrics", {}) or {}
        params = data.get("params", {}) or {}
        y_real = data.get("y_real")
        y_pred = data.get("y_pred")
        vip = data.get("vip")
        top_vips = data.get("top_vips")
        interpretacao_vips = data.get("interpretacao_vips")
        resumo_interpretativo = data.get("resumo_interpretativo")
        scores = data.get("scores")
        analysis_type = params.get("analysis_type", "PLS-R")

        # Caminhos temporários para limpar ao final
        temp_paths: list[str] = []
        scatter = None
        conf_path = None
        class_report_path = None
        vip_path = None

        # Figuras específicas por tipo de análise
        if analysis_type == "PLS-DA" and scores and y_real is not None and y_pred is not None:
            scatter = _scores_plot(scores, y_real)  # retorna path
            temp_paths.append(scatter)

            if isinstance(params.get("class_mapping"), dict):
                labels = [v for k, v in sorted(params["class_mapping"].items(), key=lambda x: int(x[0]))]
            else:
                labels = list(np.unique(y_real))
            if y_real and y_pred:
                conf_path = _cm_plot(y_real, y_pred, labels)
                temp_paths.append(conf_path)

            if metrics.get("ClassificationReport"):
                class_report_path = _class_report_plot(metrics["ClassificationReport"])
                temp_paths.append(class_report_path)
        else:
            # Regressão: scatter y_real vs y_pred
            if y_real and y_pred:
                scatter = _scatter_plot(y_real, y_pred)
                temp_paths.append(scatter)

        # VIP bar plot (se houver VIPs)
        if vip:
            vip_path = _vip_plot(vip)
            temp_paths.append(vip_path)

        # Monta PDF
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

        # Gera bytes do PDF
        pdf_bytes = pdf.pdf.output(dest="S").encode("latin1")

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="report.pdf"'}
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    finally:
        # Limpeza dos arquivos temporários gerados para o relatório
        for p in list(set([p for p in temp_paths if p])):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass


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

