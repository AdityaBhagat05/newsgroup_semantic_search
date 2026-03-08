import os
import json
import pickle
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from vectordb import (
    get_chroma_client,
    get_or_create_collection,
    embed_query,
    query_similar,
)
from clustering import predict_query_clusters, load_cluster_results
from cache import SemanticCache

class AppState:
    collection = None
    pca = None
    gmm = None
    cache: Optional[SemanticCache] = None


state = AppState()

DEFAULT_THRESHOLD = float(os.getenv("CACHE_THRESHOLD", "0.85"))
GMM_MODEL_PATH = os.path.join(os.path.dirname(__file__), "../data/gmm_model.pkl")


@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Starting up — loading resources...")

    # 1. ChromaDB
    client = get_chroma_client()
    state.collection = get_or_create_collection(client)
    n_docs = state.collection.count()
    if n_docs == 0:
        raise RuntimeError(
            "ChromaDB is empty. Run: python vectordb.py first."
        )
    print(f"✓ ChromaDB ready — {n_docs} documents")

    cluster_results = load_cluster_results()
    if cluster_results is None:
        raise RuntimeError(
            "No cluster model found. Run: python clustering.py first."
        )
    _, _, state.pca, state.gmm = cluster_results
    print(f"✓ GMM loaded — {state.gmm.n_components} clusters")

    state.cache = SemanticCache(
        similarity_threshold=DEFAULT_THRESHOLD,
        top_clusters_to_scan=2,
    )
    print(f"✓ Semantic cache ready (θ={DEFAULT_THRESHOLD})")

    print("✅ Service ready.\n")
    yield

    print("Shutting down.")


app = FastAPI(
    title="Newsgroups Semantic Search",
    description="Fuzzy-clustered semantic search with cluster-aware cache",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, example="What are the health effects of smoking?")


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: Optional[str]
    similarity_score: Optional[float]
    result: str
    dominant_cluster: int


class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float


class ThresholdRequest(BaseModel):
    threshold: float = Field(..., ge=0.01, le=1.0)


def _compute_result(query_embedding: np.ndarray, cluster_probs: np.ndarray) -> str:

    dominant_cluster = int(np.argmax(cluster_probs))

    results = query_similar(
        query_embedding,
        state.collection,
        n_results=5,
        where={"dominant_cluster": dominant_cluster} if dominant_cluster is not None else None,
    )

    if not results["ids"][0] or len(results["ids"][0]) < 3:
        results = query_similar(query_embedding, state.collection, n_results=5)

    docs = results["documents"][0]
    distances = results["distances"][0]  
    metas = results["metadatas"][0]

    lines = []
    for doc, dist, meta in zip(docs, distances, metas):
        similarity = 1 - dist 
        subject = meta.get("subject", "")
        newsgroup = meta.get("newsgroup", "")
        excerpt = doc[:300].replace("\n", " ")
        lines.append(
            f"[{newsgroup}] {subject} (sim={similarity:.3f})\n  {excerpt}..."
        )

    return "\n\n".join(lines)


@app.post("/query", response_model=QueryResponse, summary="Semantic search with cache")
async def post_query(body: QueryRequest):

    query = body.query.strip()

    query_embedding = embed_query(query)

    cluster_probs = predict_query_clusters(query_embedding, state.pca, state.gmm)
    dominant_cluster = int(np.argmax(cluster_probs))

    cache_result = state.cache.get(query, query_embedding, cluster_probs)

    if cache_result is not None:
        entry, sim_score = cache_result
        return QueryResponse(
            query=query,
            cache_hit=True,
            matched_query=entry.query,
            similarity_score=round(sim_score, 4),
            result=entry.result,
            dominant_cluster=entry.dominant_cluster,
        )

    result = _compute_result(query_embedding, cluster_probs)

    state.cache.put(query, query_embedding, cluster_probs, result)

    return QueryResponse(
        query=query,
        cache_hit=False,
        matched_query=None,
        similarity_score=None,
        result=result,
        dominant_cluster=dominant_cluster,
    )


@app.get("/cache/stats", response_model=CacheStatsResponse, summary="Cache statistics")
async def get_cache_stats():
    s = state.cache.stats()
    return CacheStatsResponse(
        total_entries=s["total_entries"],
        hit_count=s["hit_count"],
        miss_count=s["miss_count"],
        hit_rate=s["hit_rate"],
    )


@app.delete("/cache", summary="Flush cache and reset stats")
async def delete_cache():
    state.cache.flush()
    return {"status": "ok", "message": "Cache flushed and stats reset."}



@app.post("/cache/threshold", summary="Update similarity threshold at runtime")
async def set_threshold(body: ThresholdRequest):

    old = state.cache.threshold
    state.cache.set_threshold(body.threshold)
    return {"old_threshold": old, "new_threshold": body.threshold}


@app.post("/cache/explore", summary="Show threshold behaviour for a query")
async def explore_cache_threshold(body: QueryRequest):

    query_embedding = embed_query(body.query)
    cluster_probs = predict_query_clusters(query_embedding, state.pca, state.gmm)
    return state.cache.explore_threshold(query_embedding, cluster_probs)


@app.get("/health", summary="Health check")
async def health():
    return {
        "status": "healthy",
        "db_docs": state.collection.count() if state.collection else 0,
        "cache_entries": len(state.cache) if state.cache else 0,
        "n_clusters": state.gmm.n_components if state.gmm else None,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)