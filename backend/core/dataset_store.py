from typing import Any, Dict
import uuid


class DatasetStore:
    """Simple in-memory store for temporary dataset objects."""

    def __init__(self) -> None:
        self._mem: Dict[str, dict] = {}

    def save(self, dataset_id: str, payload: dict):
        self._mem[dataset_id] = payload

    def put(self, X, y_df, meta: dict | None) -> str:
        dsid = uuid.uuid4().hex
        payload = {"X": X, "y_df": y_df}
        if meta:
            payload.update(meta)
        self.save(dsid, payload)
        return dsid

    def get(self, dsid: str) -> dict:
        return self._mem.get(dsid, {})

    def has(self, dsid: str) -> bool:
        return dsid in self._mem

    def get_target(self, dataset_id: str, target_name: str):
        ds = self.get(dataset_id)
        ydf = ds.get("y_df")
        if ydf is None:
            return None
        from utils.targets import pick_column_ci, normalize_series_for_target
        col = pick_column_ci(ydf, target_name)
        if col is None:
            return None
        s = normalize_series_for_target(ydf[col])
        return s.to_numpy()


STORE = DatasetStore()
