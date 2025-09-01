from __future__ import annotations
import pandas as pd
from typing import Dict, Any, Optional
from utils.targets import pick_column_ci, normalize_series_for_target

class DatasetStore:
    def __init__(self):
        self._data: Dict[str, Dict[str, Any]] = {}

    def save(self, dataset_id: str, payload: Dict[str, Any]):
        self._data[dataset_id] = payload   # não alterar tipos aqui

    def get(self, dataset_id: str) -> Dict[str, Any]:
        return self._data.get(dataset_id, {})

    def get_target(self, dataset_id: str, target_name: str):
        ds = self.get(dataset_id)
        ydf: Optional[pd.DataFrame] = ds.get("y_df")
        if ydf is None: return None
        col = pick_column_ci(ydf, target_name)
        if col is None: return None
        s = normalize_series_for_target(ydf[col])
        return s.to_numpy()               # dtype=object quando houver strings
