from typing import Any, Dict, Tuple
import uuid


class DatasetStore:
    """Simple in-memory store for temporary dataset objects."""

    def __init__(self) -> None:
        self._mem: Dict[str, Tuple[Any, Any, dict]] = {}

    def put(self, X, y, meta: dict | None) -> str:
        dsid = uuid.uuid4().hex
        self._mem[dsid] = (X, y, meta or {})
        return dsid

    def get(self, dsid: str):
        return self._mem.get(dsid)

    def has(self, dsid: str) -> bool:
        return dsid in self._mem

    def get_target(self, dataset_id: str, target_name: str):
        ds = self._mem.get(dataset_id)
        if not ds:
            return None
        _, ydf, _ = ds
        if ydf is None or target_name not in getattr(ydf, "columns", []):
            return None
        return ydf[target_name].to_numpy().tolist()


STORE = DatasetStore()
