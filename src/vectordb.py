"""
vectordb.py
-----------
Prepares the corpus for semantic analysis and persists it for efficient retrieval.

Design decisions (Embedding & Vector Store):
  - EMBEDDING MODEL: all-MiniLM-L6-v2
    My choice of embedding model is all-MiniLM-L6-v2. We need a model that captures 
    semantic meaning accurately enough for fuzzy clustering but is lightweight enough 
    to run locally without a GPU. It outputs a 384-dimensional vector which keeps our 
    memory footprint extremely low while maintaining high performance.

  - VECTOR STORE: Custom Numpy Implementation
    I decided to build a custom vector store from scratch using numpy rather than 
    pulling in a heavy dependency. This aligns perfectly with the first principles 
    spirit of the assignment. A custom store gives complete control over the retrieval 
    math and eliminates unnecessary overhead.
"""

import os
import json
import pickle
import numpy as np
from typing import Optional
from sentence_transformers import SentenceTransformer

from data_processor import NewsDocument

MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 128
DB_DIR = os.path.join(os.path.dirname(__file__), "../data/vector_db")
EMBEDDINGS_PATH = os.path.join(DB_DIR, "embeddings.npz")
METADATA_PATH = os.path.join(DB_DIR, "metadata.json")

os.makedirs(DB_DIR, exist_ok=True)

_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model: {MODEL_NAME} ...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


class VectorStore:
    def __init__(self):
        self.ids: list[str] = []
        self.embeddings: Optional[np.ndarray] = None 
        self.documents: list[str] = []
        self.metadatas: list[dict] = []

    def count(self) -> int:
        return len(self.ids)

    def upsert(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        documents: list[str],
        metadatas: list[dict],
    ):
        existing = {id_: i for i, id_ in enumerate(self.ids)}
        for i, (doc_id, emb, doc, meta) in enumerate(
            zip(ids, embeddings, documents, metadatas)
        ):
            if doc_id in existing:
                idx = existing[doc_id]
                self.embeddings[idx] = emb
                self.documents[idx] = doc
                self.metadatas[idx] = meta
            else:
                self.ids.append(doc_id)
                self.documents.append(doc)
                self.metadatas.append(meta)
                if self.embeddings is None:
                    self.embeddings = emb.reshape(1, -1)
                else:
                    self.embeddings = np.vstack([self.embeddings, emb.reshape(1, -1)])

    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        where: Optional[dict] = None,
    ) -> dict:
        """
        Return top-n most similar documents.
        `where`: dict of metadata key-value filters applied before similarity search.
        """
        if self.embeddings is None or len(self.ids) == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        if where:
            candidates = [
                i for i, m in enumerate(self.metadatas)
                if all(m.get(k) == v for k, v in where.items())
            ]
        else:
            candidates = list(range(len(self.ids)))

        if not candidates:
            candidates = list(range(len(self.ids)))  

        candidate_embeddings = self.embeddings[candidates]  
        
        # Because I normalize the embeddings at the time of encoding (making them unit vectors), 
        # I can calculate cosine similarity using a simple and highly optimized dot product 
        # operation via numpy matrix multiplication.
        sims = candidate_embeddings @ query_embedding 

        top_k = min(n_results, len(candidates))
        top_local = np.argsort(sims)[::-1][:top_k]
        top_global = [candidates[j] for j in top_local]

        return {
            "ids": [[self.ids[j] for j in top_global]],
            "documents": [[self.documents[j] for j in top_global]],
            "metadatas": [[self.metadatas[j] for j in top_global]],
            "distances": [[float(1 - sims[j]) for j in top_local]],  
        }

    def get(self, ids: Optional[list[str]] = None) -> dict:
        """Retrieve entries by id (or all if ids is None)."""
        if ids is None:
            return {
                "ids": self.ids,
                "embeddings": self.embeddings,
                "metadatas": self.metadatas,
            }
        idx_map = {id_: i for i, id_ in enumerate(self.ids)}
        indices = [idx_map[id_] for id_ in ids if id_ in idx_map]
        return {
            "ids": [self.ids[i] for i in indices],
            "embeddings": self.embeddings[indices] if self.embeddings is not None else [],
            "metadatas": [self.metadatas[i] for i in indices],
        }

    def update_metadata(self, ids: list[str], metadatas: list[dict]):
        """Merge new metadata fields into existing entries."""
        idx_map = {id_: i for i, id_ in enumerate(self.ids)}
        for doc_id, new_meta in zip(ids, metadatas):
            if doc_id in idx_map:
                self.metadatas[idx_map[doc_id]].update(new_meta)

    def save(self):
        """
        Persist to disk.
        State management is handled by persisting the numpy arrays to compressed .npz files 
        and the metadata to standard JSON. This ensures the FastAPI service will start 
        cleanly and instantly on subsequent runs without needing to recompute the entire corpus.
        """
        np.savez_compressed(EMBEDDINGS_PATH, embeddings=self.embeddings)
        with open(METADATA_PATH, "w") as f:
            json.dump(
                {"ids": self.ids, "documents": self.documents, "metadatas": self.metadatas},
                f,
            )
        print(f"Vector store saved ({len(self.ids)} docs) -> {DB_DIR}")

    @classmethod
    def load(cls) -> "VectorStore":
        """Load from disk."""
        vs = cls()
        if not os.path.exists(EMBEDDINGS_PATH) or not os.path.exists(METADATA_PATH):
            return vs
        data = np.load(EMBEDDINGS_PATH)
        vs.embeddings = data["embeddings"].astype(np.float32)
        with open(METADATA_PATH) as f:
            meta = json.load(f)
        vs.ids = meta["ids"]
        vs.documents = meta["documents"]
        vs.metadatas = meta["metadatas"]
        print(f"Vector store loaded ({len(vs.ids)} docs)")
        return vs


_store: Optional[VectorStore] = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore.load()
    return _store


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a list of texts, returns (N, 384) float32 array."""
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True, # Forces unit vectors for downstream dot-product similarity
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    model = get_model()
    vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
    return vec[0].astype(np.float32)


def build_vector_db(docs: list[NewsDocument], batch_size: int = 500) -> VectorStore:
    store = get_store()
    if store.count() >= len(docs):
        print(f"Vector store already contains {store.count()} documents. Skipping.")
        return store

    print(f"Embedding {len(docs)} documents in batches of {batch_size}...")
    for i in range(0, len(docs), batch_size):
        batch = docs[i: i + batch_size]
        texts = [d.full_text for d in batch]
        embeddings = embed_texts(texts)
        store.upsert(
            ids=[d.doc_id for d in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {"newsgroup": d.newsgroup, "subject": d.subject, "word_count": d.word_count}
                for d in batch
            ],
        )
        print(f"  Batch {i // batch_size + 1}/{(len(docs) - 1) // batch_size + 1} done")

    store.save()
    print(f"Vector DB ready. Total documents: {store.count()}")
    return store


def query_similar(
    query_embedding: np.ndarray,
    store: VectorStore,
    n_results: int = 5,
    where: Optional[dict] = None,
) -> dict:
    return store.query(query_embedding, n_results=n_results, where=where)


def get_all_embeddings(store: VectorStore) -> tuple[list[str], np.ndarray, list[dict]]:
    result = store.get()
    return result["ids"], result["embeddings"], result["metadatas"]


def update_cluster_metadata(
    store: VectorStore,
    ids: list[str],
    dominant_clusters: list[int],
    cluster_probs: list[list[float]],
):
    metadatas = [
        {
            "dominant_cluster": int(dominant_clusters[i]),
            "cluster_probs": json.dumps([round(float(p), 4) for p in cluster_probs[i]]),
        }
        for i in range(len(ids))
    ]
    store.update_metadata(ids, metadatas)
    store.save()
    print(f"Updated cluster metadata for {len(ids)} documents.")


def get_chroma_client():
    return None

def get_or_create_collection(_client):
    return get_store()


if __name__ == "__main__":
    from data_processor import load_corpus
    archive = r"../data/original/20_newsgroups.tar.gz"
    docs = load_corpus(archive)
    store = build_vector_db(docs)
    print(f"\nVector store has {store.count()} documents.")