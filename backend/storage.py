"""Persistence layer for call analysis records.

Provides a small CallStore interface with two backends:

- MongoCallStore: persists records to MongoDB (queryable, concurrency-safe).
- JsonCallStore:  persists to data/call_log.json (legacy file-based).

get_call_store() picks the backend from settings.PERSISTENCE_BACKEND. When
"mongo" is selected but the server is unreachable, it logs a warning and falls
back to the JSON store so the app still runs right after a fresh git sync
("Mongo not running" must never crash the demo).
"""

from __future__ import annotations

import json as _json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

try:
    from backend.schemas import CallRecord
    from backend.config import settings
except ModuleNotFoundError:  # run from within backend/
    from schemas import CallRecord
    from config import settings

logger = logging.getLogger(__name__)

_CALL_LOG_FILE = Path(__file__).resolve().parents[1] / "data" / "call_log.json"


def _build_notes(record: CallRecord, notes: str) -> CallRecord:
    """Return a copy of record with notes replaced."""
    return record.model_copy(update={"notes": notes})


class CallStore(ABC):
    """Storage interface for call analysis records."""

    @abstractmethod
    def add(self, record: CallRecord) -> None:
        ...

    @abstractmethod
    def list(self, limit: int) -> List[CallRecord]:
        """Most-recent-first, capped at limit."""
        ...

    @abstractmethod
    def get(self, call_id: str) -> Optional[CallRecord]:
        """Fetch a single record by call_id, or None."""
        ...

    @abstractmethod
    def update_notes(self, call_id: str, notes: str) -> bool:
        """Set notes on a record. Returns True if a record matched."""
        ...

    @abstractmethod
    def delete(self, call_id: str) -> bool:
        """Delete one record. Returns True if a record matched."""
        ...

    @abstractmethod
    def delete_all(self) -> None:
        ...

    @abstractmethod
    def count(self) -> int:
        ...

    @abstractmethod
    def latest(self) -> Optional[CallRecord]:
        ...


class JsonCallStore(CallStore):
    """File-backed store. Keeps an in-memory mirror synced to call_log.json."""

    def __init__(self, path: Path = _CALL_LOG_FILE) -> None:
        self._path = path
        self._log: List[CallRecord] = self._load()

    def _load(self) -> List[CallRecord]:
        if self._path.is_file():
            try:
                raw = _json.loads(self._path.read_text(encoding="utf-8"))
                return [CallRecord(**r) for r in raw]
            except Exception:
                return []
        return []

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            _json.dumps([r.model_dump(mode="json") for r in self._log], indent=2),
            encoding="utf-8",
        )

    def add(self, record: CallRecord) -> None:
        self._log.append(record)
        self._save()

    def list(self, limit: int) -> List[CallRecord]:
        return list(reversed(self._log))[:limit]

    def get(self, call_id: str) -> Optional[CallRecord]:
        for record in self._log:
            if record.call_id == call_id:
                return record
        return None

    def update_notes(self, call_id: str, notes: str) -> bool:
        for idx, record in enumerate(self._log):
            if record.call_id == call_id:
                self._log[idx] = _build_notes(record, notes)
                self._save()
                return True
        return False

    def delete(self, call_id: str) -> bool:
        for idx, record in enumerate(self._log):
            if record.call_id == call_id:
                self._log.pop(idx)
                self._save()
                return True
        return False

    def delete_all(self) -> None:
        self._log.clear()
        self._save()

    def count(self) -> int:
        return len(self._log)

    def latest(self) -> Optional[CallRecord]:
        return self._log[-1] if self._log else None


class MongoCallStore(CallStore):
    """MongoDB-backed store. One document per call, unique index on call_id."""

    def __init__(self, collection) -> None:
        self._col = collection
        self._col.create_index("call_id", unique=True)

    @staticmethod
    def _to_record(doc: dict) -> CallRecord:
        return CallRecord(**{k: v for k, v in doc.items() if k != "_id"})

    def add(self, record: CallRecord) -> None:
        self._col.insert_one(record.model_dump(mode="json"))

    def list(self, limit: int) -> List[CallRecord]:
        cursor = self._col.find().sort("timestamp", -1).limit(limit)
        return [self._to_record(d) for d in cursor]

    def get(self, call_id: str) -> Optional[CallRecord]:
        doc = self._col.find_one({"call_id": call_id})
        return self._to_record(doc) if doc else None

    def update_notes(self, call_id: str, notes: str) -> bool:
        result = self._col.update_one({"call_id": call_id}, {"$set": {"notes": notes}})
        return result.matched_count > 0

    def delete(self, call_id: str) -> bool:
        result = self._col.delete_one({"call_id": call_id})
        return result.deleted_count > 0

    def delete_all(self) -> None:
        self._col.delete_many({})

    def count(self) -> int:
        return self._col.count_documents({})

    def latest(self) -> Optional[CallRecord]:
        doc = self._col.find_one(sort=[("timestamp", -1)])
        return self._to_record(doc) if doc else None


def _build_mongo_store() -> MongoCallStore:
    from pymongo import MongoClient

    client = MongoClient(settings.MONGODB_URI, serverSelectionTimeoutMS=1500)
    client.admin.command("ping")  # raises if unreachable
    collection = client[settings.MONGODB_DB][settings.MONGODB_COLLECTION]
    return MongoCallStore(collection)


_store: Optional[CallStore] = None


def get_call_store() -> CallStore:
    """Return the configured store (singleton). Falls back to JSON if Mongo down."""
    global _store
    if _store is not None:
        return _store

    backend = settings.PERSISTENCE_BACKEND.lower()
    if backend == "mongo":
        try:
            _store = _build_mongo_store()
            logger.info("Persistence: MongoDB at %s", settings.MONGODB_URI)
        except Exception as exc:
            logger.warning(
                "Persistence: MongoDB unreachable (%s); falling back to JSON store.",
                exc,
            )
            _store = JsonCallStore()
    else:
        _store = JsonCallStore()
        logger.info("Persistence: JSON file store (%s)", _CALL_LOG_FILE)

    return _store
