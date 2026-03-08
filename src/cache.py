"""
cache.py
--------
Cluster-aware semantic cache — built entirely from scratch.
No Redis, Memcached, or any caching library.

Design overview:
  The cache maps query embeddings to results. Instead of exact string matching,
  it checks if an incoming query is "close enough" (by cosine similarity) to
  any cached query. If yes, it returns the cached result without recomputing.

The key tunable decision — similarity threshold (θ):
  - θ = 0.70: very permissive. "What planets orbit the sun?" would hit
    on "How many planets are in the solar system?". This inflates hit rate
    but returns wrong or misleading answers for genuinely different questions.
  - θ = 0.85: a reasonable default. It catches genuine paraphrases
    ("machine learning applications" ≈ "uses of ML") while rejecting
    questions that are merely in the same topic area.
  - θ = 0.95: extremely conservative. Almost pure deduplication.
    Useful if the result computation is very cheap and you care more about
    accuracy than latency reduction.
  
  What this reveals: At low θ, the cache reveals which semantic neighborhoods 
  are densely queried. At high θ, it reveals exact query repetition patterns.

Cluster-aware lookup (the O(N) efficiency trick):
  A naïve cache scans all N entries to find similar queries, which is O(N) per lookup.
  By using the cluster structure built in Part 2, I first identify the dominant 
  cluster(s) of the new query, then only scan cached entries from those specific clusters. 
  If the cache has entries across K clusters and I only scan the top-2 clusters, 
  I reduce expected scan cost from O(N) to O(N/K * 2). This is a massive speedup as N grows.
"""

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
        
        # We scan the top 2 clusters to account for "boundary" queries that straddle topics
        self.top_k_clusters = top_clusters_to_scan

        # Primary Data Structure: A dictionary mapping Cluster ID to a list of CacheEntries.
        # This enables O(1) cluster-bucket lookup.
        self._cluster_buckets: dict[int, list[CacheEntry]] = defaultdict(list)

        # Fast-path global index: Maps a SHA256 query hash to a CacheEntry.
        # This allows identical queries to bypass the similarity scan entirely for an O(1) hit.
        self._exact_index: dict[str, CacheEntry] = {}

        self._hit_count = 0
        self._miss_count = 0

        # Thread safety via RLock allows multiple FastAPI workers to safely share the cache instance
        self._lock = threading.RLock()


    def get(
        self,
        query: str,
        query_embedding: np.ndarray,
        cluster_probs: np.ndarray,
    ) -> Optional[tuple[CacheEntry, float]]:
        with self._lock:
            # Step 1: Exact match fast-path (O(1) lookup)
            qhash = _hash_query(query)
            if qhash in self._exact_index:
                entry = self._exact_index[qhash]
                entry.hit_count += 1
                self._hit_count += 1
                return entry, 1.0 

            # Step 2: Cluster-aware similarity scan (O(N/K) lookup)
            # Find the top-K clusters by this query's membership probability
            top_cluster_ids = np.argsort(cluster_probs)[::-1][: self.top_k_clusters]

            best_entry = None
            best_sim = -1.0

            # Only scan the buckets belonging to the query's dominant clusters
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
        # Assign the new cache entry to its dominant cluster bucket
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
    """
    Since embeddings are normalized at encoding time (normalize_embeddings=True),
    cosine similarity reduces to a simple, highly optimized dot product.
    """
    return float(np.dot(a, b))


def _hash_query(query: str) -> str:
    normalised = query.strip().lower()
    return hashlib.sha256(normalised.encode()).hexdigest()