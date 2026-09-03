"""
backend/app/services/embed_service.py
=======================================
AbuseRing Sentinel — HuggingFace Embedding Service

Calls the HuggingFace Inference API to embed text using
sentence-transformers/all-MiniLM-L6-v2 (384-dim vectors).

Falls back to a zero vector if the API is unavailable.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import requests

log = logging.getLogger(__name__)

_HF_INFERENCE_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"
_TIMEOUT = 15  # seconds
_RETRIES = 3


class EmbedService:
    """Text embedding via HuggingFace Inference API."""

    def __init__(self, api_key: str, model: str, dim: int = 384):
        self._api_key = api_key
        self._model = model
        self._dim = dim
        self._url = _HF_INFERENCE_URL.format(model=model)
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def embed(self, text: str) -> list[float]:
        """
        Embed a single string. Returns a 384-dim float list.
        Retries up to 3 times with exponential backoff.
        """
        text = text[:512]  # token safety limit
        for attempt in range(_RETRIES):
            try:
                resp = requests.post(
                    self._url,
                    headers=self._headers,
                    json={"inputs": text, "options": {"wait_for_model": True}},
                    timeout=_TIMEOUT,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    # API returns list of floats (sentence embedding)
                    if isinstance(data, list) and isinstance(data[0], float):
                        return data
                    # Some models return [[floats]] — mean pool
                    if isinstance(data, list) and isinstance(data[0], list):
                        vecs = data[0]
                        if isinstance(vecs[0], list):
                            # token-level embeddings → mean pool
                            import numpy as np
                            return list(np.mean(vecs, axis=0).tolist())
                        return vecs
                log.warning(f"  HF embed attempt {attempt+1} failed: {resp.status_code} {resp.text[:100]}")
            except Exception as e:
                log.warning(f"  HF embed attempt {attempt+1} exception: {e}")
            time.sleep(2 ** attempt)

        log.error("  HF embedding failed after all retries — returning zero vector")
        return [0.0] * self._dim

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def health_check(self) -> bool:
        try:
            vec = self.embed("health check")
            return len(vec) == self._dim
        except Exception:
            return False
