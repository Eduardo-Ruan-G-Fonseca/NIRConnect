"""Thread-safe in-memory dataset storage.

This module provides a very small utility used by the API endpoints to
temporarily persist uploaded datasets.  Each dataset is stored as a
``pandas.DataFrame`` and referenced by a random UUID.  The store behaves like
an in-memory LRU cache – recent datasets are kept while older entries can be
evicted to prevent unbounded memory growth.  For the current use cases a very
small cache is enough, therefore the implementation below keeps the last
``MAX_ITEMS`` datasets.

The interface intentionally mirrors a tiny subset of what would be provided by
an external cache (Redis, Memcached, …).  Only two operations are required:

``put(df)``
    Store the given DataFrame and return the generated ``dataset_id``.

``get(dataset_id)``
    Retrieve a previously stored DataFrame or ``None`` when the id is
    unknown/expired.

The object is implemented as a singleton so multiple modules can import and
use ``DatasetStore()`` without explicitly sharing state.  All operations are
protected by a re-entrant lock making them safe to use from async endpoints
running in different threads.
"""

from __future__ import annotations

from collections import OrderedDict
import threading
import uuid
from typing import Optional

import pandas as pd


class DatasetStore:
    """Singleton, thread-safe cache of uploaded datasets."""

    # Rough safeguard to avoid holding too many large DataFrames in memory.
    MAX_ITEMS = 8

    _instance: "DatasetStore | None" = None
    _instance_lock = threading.Lock()

    def __new__(cls) -> "DatasetStore":
        # Standard singleton pattern with double-checked locking
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._data = OrderedDict()  # type: ignore[attr-defined]
                    cls._instance._lock = threading.RLock()  # type: ignore[attr-defined]
        return cls._instance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def put(self, df: pd.DataFrame) -> str:
        """Store ``df`` and return a new UUID string."""

        dataset_id = uuid.uuid4().hex
        with self._lock:  # type: ignore[attr-defined]
            self._data[dataset_id] = df  # type: ignore[index]
            self._data.move_to_end(dataset_id)  # LRU behaviour
            # Drop the oldest item if cache grows too big
            if len(self._data) > self.MAX_ITEMS:  # type: ignore[attr-defined]
                self._data.popitem(last=False)
        return dataset_id

    def get(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Return the DataFrame associated with ``dataset_id`` or ``None``."""

        with self._lock:  # type: ignore[attr-defined]
            df = self._data.get(dataset_id)  # type: ignore[attr-defined]
            if df is not None:
                # Touch item to keep it as most recently used
                self._data.move_to_end(dataset_id)  # type: ignore[attr-defined]
        return df


# Convenience function mirroring a typical ``get_instance`` API
def get_store() -> DatasetStore:
    return DatasetStore()


__all__ = ["DatasetStore", "get_store"]

