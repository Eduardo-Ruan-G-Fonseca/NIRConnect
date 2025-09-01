import uuid
from typing import Any, Dict, Optional

import pandas as pd

from utils.targets import pick_column_ci, normalize_series_for_target


class DatasetStore:
    """Simple in-memory store for temporary dataset objects."""

    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, Any]] = {}

    def save(self, dataset_id: str, payload: Dict[str, Any]):
        """Persist *payload* exactly as provided."""
        self._data[dataset_id] = payload

    def put(self, X, y_df, meta: dict | None) -> str:
        dsid = uuid.uuid4().hex
        payload = {"X": X, "y_df": y_df}
        if meta:
            payload.update(meta)
        self.save(dsid, payload)
        return dsid

    def get(self, dataset_id: str) -> Dict[str, Any]:
        return self._data.get(dataset_id, {})

    def has(self, dataset_id: str) -> bool:
        return dataset_id in self._data

    def get_target(self, dataset_id: str, target_name: str):
        ds = self.get(dataset_id)
        ydf: Optional[pd.DataFrame] = ds.get("y_df")
        if ydf is None:
            return None
        col = pick_column_ci(ydf, target_name)
        if col is None:
            return None
        # Preserve original dtype: do not force float conversion
        s = normalize_series_for_target(ydf[col])
        return s.to_numpy()  # dtype=object when there are strings


STORE = DatasetStore()
