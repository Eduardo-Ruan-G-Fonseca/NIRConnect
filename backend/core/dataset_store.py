from typing import Optional
from threading import Lock
import pandas as pd
import uuid


class DatasetStore:
    _instance = None
    _lock = Lock()

    def __init__(self):
        self._by_id: dict[str, pd.DataFrame] = {}
        self._last_id: Optional[str] = None

    @classmethod
    def inst(cls):
        with cls._lock:
            if not cls._instance:
                cls._instance = DatasetStore()
            return cls._instance

    def put(self, df: pd.DataFrame) -> str:
        did = str(uuid.uuid4())
        self._by_id[did] = df
        self._last_id = did
        return did

    def get(self, did: Optional[str]) -> Optional[pd.DataFrame]:
        if did and did in self._by_id:
            return self._by_id[did]
        if self._last_id:
            return self._by_id.get(self._last_id)
        return None
