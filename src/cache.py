import time
import hashlib
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Any
from collections import defaultdict


@dataclass
class CacheEntry:
    query: str
    query_embedding: np.ndarray
    result: Any
    dominant_cluster: int
    cluster_probs: np.ndarray  
    timestamp: float = field(default_factory=time.time)
    hit_count: int = 0  


class SemanticCache:

    def __init__(self, similarity_threshold: float = 0.85, top_clusters_to_scan: int = 2):

        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1]")

        self.threshold = similarity_threshold
        self.top_k_clusters = top_clusters_to_scan

        self._cluster_buckets: dict[int, list[CacheEntry]] = defaultdict(list)

        self._exact_index: dict[str, CacheEntry] = {}

        self._hit_count = 0
        self._miss_count = 0

        self._lock = threading.RLock()


    def get(
        self,
        query: str,
        query_embedding: np.ndarray,
        cluster_probs: np.ndarray,
    ) -> Optional[tuple[CacheEntry, float]]:
        with self._lock:
            qhash = _hash_query(query)
            if qhash in self._exact_index:
                entry = self._exact_index[qhash]
                entry.hit_count += 1
                self._hit_count += 1
                return entry, 1.0 

            top_cluster_ids = np.argsort(cluster_probs)[::-1][: self.top_k_clusters]

            best_entry = None
            best_sim = -1.0

            for cluster_id in top_cluster_ids:
                bucket = self._cluster_buckets.get(int(cluster_id), [])
                for entry in bucket:
                    sim = _cosine_similarity(query_embedding, entry.query_embedding)
                    if sim > best_sim:
                        best_sim = sim
                        best_entry = entry

            if best_sim >= self.threshold and best_entry is not None:
                best_entry.hit_count += 1
                self._hit_count += 1
                return best_entry, float(best_sim)

            self._miss_count += 1
            return None

    def put(
        self,
        query: str,
        query_embedding: np.ndarray,
        cluster_probs: np.ndarray,
        result: Any,
    ) -> CacheEntry:
        dominant_cluster = int(np.argmax(cluster_probs))
        entry = CacheEntry(
            query=query,
            query_embedding=query_embedding.copy(),
            result=result,
            dominant_cluster=dominant_cluster,
            cluster_probs=cluster_probs.copy(),
        )

        with self._lock:
            self._cluster_buckets[dominant_cluster].append(entry)
            self._exact_index[_hash_query(query)] = entry

        return entry

    def stats(self) -> dict:
        with self._lock:
            total = self._hit_count + self._miss_count
            return {
                "total_entries": sum(len(b) for b in self._cluster_buckets.values()),
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": round(self._hit_count / total, 4) if total > 0 else 0.0,
                "threshold": self.threshold,
                "cluster_distribution": {
                    str(k): len(v) for k, v in self._cluster_buckets.items()
                },
            }

    def flush(self):
        """Clear all cache entries and reset stats."""
        with self._lock:
            self._cluster_buckets.clear()
            self._exact_index.clear()
            self._hit_count = 0
            self._miss_count = 0

    def set_threshold(self, new_threshold: float):
        if not 0.0 < new_threshold <= 1.0:
            raise ValueError("Threshold must be in (0, 1]")
        with self._lock:
            self.threshold = new_threshold

    def explore_threshold(
        self,
        query_embedding: np.ndarray,
        cluster_probs: np.ndarray,
        thresholds: list[float] = None,
    ) -> dict:
        if thresholds is None:
            thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

        top_cluster_ids = np.argsort(cluster_probs)[::-1][: self.top_k_clusters]
        candidates = []

        with self._lock:
            for cluster_id in top_cluster_ids:
                for entry in self._cluster_buckets.get(int(cluster_id), []):
                    sim = _cosine_similarity(query_embedding, entry.query_embedding)
                    candidates.append((sim, entry.query))

        candidates.sort(reverse=True)
        return {
            "top_candidates": [
                {"query": q, "similarity": round(s, 4)} for s, q in candidates[:10]
            ],
            "threshold_analysis": {
                str(θ): sum(1 for s, _ in candidates if s >= θ) for θ in thresholds
            },
        }

    def __len__(self) -> int:
        with self._lock:
            return sum(len(b) for b in self._cluster_buckets.values())

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"SemanticCache(entries={s['total_entries']}, "
            f"hit_rate={s['hit_rate']:.2%}, θ={self.threshold})"
        )



def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _hash_query(query: str) -> str:
    normalised = query.strip().lower()
    return hashlib.sha256(normalised.encode()).hexdigest()