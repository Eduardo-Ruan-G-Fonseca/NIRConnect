# -*- coding: utf-8 -*-
from __future__ import annotations
import os, uuid, joblib, threading
from typing import Any, Dict, Optional, Tuple

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(_MODELS_DIR, exist_ok=True)

_LOCK = threading.Lock()
_DATA_CACHE: Dict[str, Tuple[Any, Any, Dict[str, Any]]] = {}
_LAST_DATA_ID: Optional[str] = None

def store_dataset(X, y, meta: Optional[Dict[str, Any]] = None) -> str:
    """Guarda X, y e metadados em cache e em disco leve, retorna data_id."""
    global _LAST_DATA_ID
    data_id = str(uuid.uuid4())
    with _LOCK:
        _DATA_CACHE[data_id] = (X, y, meta or {})
        _LAST_DATA_ID = data_id
    path = os.path.join(_MODELS_DIR, f"{data_id}.joblib")
    joblib.dump({"X": X, "y": y, "meta": meta or {}}, path)
    return data_id

def resolve_dataset(data_id: Optional[str]) -> Tuple[Any, Any, Dict[str, Any], str]:
    """
    Resolve (X,y,meta,id) por:
      1) data_id informado;
      2) último data_id em cache (_LAST_DATA_ID);
      3) tentativa de rehidratar do disco se houver último id.
    Lança ValueError com mensagem amigável se não houver dataset.
    """
    global _LAST_DATA_ID
    with _LOCK:
        candidate = data_id or _LAST_DATA_ID
    if candidate and candidate in _DATA_CACHE:
        X, y, meta = _DATA_CACHE[candidate]
        return X, y, meta, candidate

    if not data_id and _LAST_DATA_ID:
        path = os.path.join(_MODELS_DIR, f"{_LAST_DATA_ID}.joblib")
        if os.path.exists(path):
            blob = joblib.load(path)
            with _LOCK:
                _DATA_CACHE[_LAST_DATA_ID] = (blob["X"], blob["y"], blob.get("meta", {}))
            X, y, meta = _DATA_CACHE[_LAST_DATA_ID]
            return X, y, meta, _LAST_DATA_ID

    if data_id:
        path = os.path.join(_MODELS_DIR, f"{data_id}.joblib")
        if os.path.exists(path):
            blob = joblib.load(path)
            with _LOCK:
                _DATA_CACHE[data_id] = (blob["X"], blob["y"], blob.get("meta", {}))
            X, y, meta = _DATA_CACHE[data_id]
            with _LOCK:
                _LAST_DATA_ID = data_id
            return X, y, meta, data_id

    raise ValueError("Nenhum dataset está carregado. Execute a etapa de análise/preparo para gerar um data_id e reenvie com ele.")
