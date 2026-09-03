"""
backend/app/db/qdrant.py
========================
AbuseRing Sentinel — Qdrant Cloud Vector DB Client

Manages the 'argus-mesh-vectors' collection (384-dim, cosine distance)
used for semantic search over historical investigation cases.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient as _QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

log = logging.getLogger(__name__)


class QdrantDB:
    """Qdrant Cloud client for AbuseRing Sentinel case vectors."""

    def __init__(self, url: str, api_key: str, collection: str, dim: int = 384):
        self._url = url
        self._api_key = api_key
        self._collection = collection
        self._dim = dim
        self._client: _QdrantClient | None = None

    def connect(self) -> None:
        log.info(f"Connecting to Qdrant: {self._url}")
        self._client = _QdrantClient(url=self._url, api_key=self._api_key)
        log.info("  ✅ Qdrant connected")

    def close(self) -> None:
        # qdrant_client does not require explicit close
        log.info("Qdrant client released")

    def ensure_collection(self) -> None:
        """Create the collection if it does not exist."""
        if not self._client:
            raise RuntimeError("Qdrant not connected")
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
            )
            log.info(f"  ✅ Qdrant collection created: {self._collection}")
        else:
            log.info(f"  Qdrant collection already exists: {self._collection}")

    def count(self) -> int:
        if not self._client:
            return 0
        return self._client.count(self._collection).count

    def upsert(self, point_id: str, vector: list[float], payload: dict) -> None:
        """Upsert a single point into the collection."""
        if not self._client:
            raise RuntimeError("Qdrant not connected")
        # Use a deterministic UUID derived from point_id string
        uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, point_id))
        self._client.upsert(
            collection_name=self._collection,
            points=[PointStruct(id=uid, vector=vector, payload=payload)],
        )

    def search(
        self,
        vector: list[float],
        top_k: int = 3,
        score_threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Return top_k similar points with score and payload."""
        if not self._client:
            raise RuntimeError("Qdrant not connected")
        results = self._client.search(
            collection_name=self._collection,
            query_vector=vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [
            {"score": r.score, **r.payload}
            for r in results
        ]

    def get_all(self, limit: int = 100) -> list[dict]:
        """Scroll through all points (for listing stored cases)."""
        if not self._client:
            return []
        records, _ = self._client.scroll(
            collection_name=self._collection,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [r.payload for r in records]

    def health_check(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception as e:
            log.error(f"Qdrant health check failed: {e}")
            return False
