"""
clustering.py
-------------
Fuzzy (soft) clustering of the 20 Newsgroups corpus.

Design decisions (Clustering):
  - WHY GMM OVER K-MEANS: The assignment explicitly forbids hard cluster 
    assignments. A document about gun legislation belongs to both politics 
    and firearms. K-Means forces a single label, but a Gaussian Mixture Model (GMM) 
    returns a probability distribution across all clusters.
  - WHY PCA FIRST: Fitting a GMM in 384 dimensions requires estimating a massive 
    covariance matrix, which is highly unstable with only ~18k documents. 
    I project down to 100 components using PCA first to make the math stable and fast.
  - NUMBER OF CLUSTERS: I don't guess the number of clusters. I test a range 
    from 10 to 22 and use the Bayesian Information Criterion (BIC) to find the optimal K. 
    BIC penalizes model complexity, so it mathematically identifies where adding 
    more clusters stops providing meaningful semantic separation.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from collections import defaultdict
from typing import Optional

from vectordb import get_store, get_all_embeddings, update_cluster_metadata

PCA_COMPONENTS = 100  
K_RANGE = range(10, 22)
GMM_COVARIANCE_TYPE = "diag"  
GMM_MAX_ITER = 200
RANDOM_STATE = 42

CLUSTER_DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/cluster_model.npz")
GMM_MODEL_PATH = os.path.join(os.path.dirname(__file__), "../data/gmm_model.pkl")


def reduce_dimensions(embeddings: np.ndarray, n_components: int = PCA_COMPONENTS) -> tuple[np.ndarray, PCA]:
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    reduced = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA: {n_components} components explain {explained:.1%} of variance.")
    return reduced, pca


def select_n_clusters(reduced_embeddings: np.ndarray, k_range=K_RANGE) -> int:
    print(f"Selecting optimal K via BIC over range {list(k_range)}...")
    bics = []
    for k in k_range:
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=GMM_COVARIANCE_TYPE,
            max_iter=GMM_MAX_ITER,
            random_state=RANDOM_STATE,
            n_init=3,  
        )
        gmm.fit(reduced_embeddings)
        # Calculate BIC to balance model fit against complexity
        bic = gmm.bic(reduced_embeddings)
        bics.append(bic)
        print(f"  K={k:2d}  BIC={bic:.0f}")

    bic_arr = np.array(bics)
    if len(bic_arr) >= 3:
        second_deriv = np.diff(bic_arr, n=2)
        elbow_idx = int(np.argmax(second_deriv)) + 1
    else:
        elbow_idx = int(np.argmin(bic_arr))
    optimal_k = list(k_range)[elbow_idx]

    plt.figure(figsize=(8, 4))
    plt.plot(list(k_range), bics, 'o-', color='steelblue')
    plt.axvline(optimal_k, color='red', linestyle='--', label=f'Optimal K={optimal_k}')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('BIC Score (lower = better)')
    plt.title('GMM Cluster Selection via BIC')
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "../data/bic_curve.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"BIC curve saved to {plot_path}")
    print(f"→ Optimal K = {optimal_k}")
    return optimal_k


def fit_gmm(reduced_embeddings: np.ndarray, n_clusters: int) -> GaussianMixture:
    print(f"Fitting GMM with K={n_clusters}...")
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type=GMM_COVARIANCE_TYPE,
        max_iter=GMM_MAX_ITER,
        random_state=RANDOM_STATE,
        n_init=5,  
        verbose=1,
    )
    gmm.fit(reduced_embeddings)
    print(f"GMM converged: {gmm.converged_}")
    return gmm


def analyse_clusters(
    gmm: GaussianMixture,
    probs: np.ndarray,
    metadatas: list[dict],
    ids: list[str],
) -> dict:
    n_clusters = gmm.n_components
    dominant_clusters = np.argmax(probs, axis=1)

    # I calculate information entropy to find boundary documents. 
    # High entropy means the model is highly uncertain because the document 
    # legitimately spans multiple clusters.
    entropy = -np.sum(probs * np.log(probs + 1e-9), axis=1)
    max_entropy = np.log(n_clusters) 
    norm_entropy = entropy / max_entropy

    profiles = {}
    for k in range(n_clusters):
        mask = dominant_clusters == k
        member_metadatas = [metadatas[i] for i in range(len(metadatas)) if mask[i]]
        member_ids = [ids[i] for i in range(len(ids)) if mask[i]]
        ng_weights = defaultdict(float)
        for i in range(len(metadatas)):
            ng = metadatas[i].get("newsgroup", "unknown")
            ng_weights[ng] += probs[i, k]

        top_newsgroups = sorted(ng_weights.items(), key=lambda x: -x[1])[:5]

        if member_ids:
            member_entropy = norm_entropy[mask]
            most_confident_idx = np.argsort(member_entropy)[:3]
            core_docs = [
                {
                    "id": member_ids[j],
                    "newsgroup": member_metadatas[j].get("newsgroup"),
                    "subject": member_metadatas[j].get("subject", ""),
                    "entropy": float(member_entropy[j]),
                }
                for j in most_confident_idx
            ]
        else:
            core_docs = []

        profiles[k] = {
            "cluster_id": k,
            "size": int(mask.sum()),
            "top_newsgroups": [(ng, round(w, 2)) for ng, w in top_newsgroups],
            "dominant_newsgroup": top_newsgroups[0][0] if top_newsgroups else "?",
            "core_docs": core_docs,
        }
        
    # Boundary threshold to isolate genuine cross topic documents
    boundary_mask = norm_entropy > 0.3
    boundary_docs = [
        {
            "id": ids[i],
            "newsgroup": metadatas[i].get("newsgroup"),
            "subject": metadatas[i].get("subject", ""),
            "entropy": float(norm_entropy[i]),
            "top_clusters": sorted(
                enumerate(probs[i].tolist()), key=lambda x: -x[1]
            )[:3],
        }
        for i in np.where(boundary_mask)[0][:20]  
    ]

    return {
        "n_clusters": n_clusters,
        "cluster_profiles": profiles,
        "boundary_documents": boundary_docs,
        "entropy_stats": {
            "mean": float(norm_entropy.mean()),
            "median": float(np.median(norm_entropy)),
            "pct_high_uncertainty": float(boundary_mask.mean()),
        },
    }


def save_cluster_results(
    ids: list[str],
    probs: np.ndarray,
    pca: PCA,
    gmm: GaussianMixture,
):
    import pickle
    np.savez_compressed(
        CLUSTER_DATA_PATH,
        ids=np.array(ids),
        probs=probs,
    )
    with open(GMM_MODEL_PATH, "wb") as f:
        pickle.dump({"gmm": gmm, "pca": pca}, f)
    print(f"Cluster data saved to {CLUSTER_DATA_PATH}")


def load_cluster_results() -> Optional[tuple]:
    import pickle
    if not os.path.exists(CLUSTER_DATA_PATH) or not os.path.exists(GMM_MODEL_PATH):
        return None
    data = np.load(CLUSTER_DATA_PATH, allow_pickle=True)
    with open(GMM_MODEL_PATH, "rb") as f:
        models = pickle.load(f)
    return (
        list(data["ids"]),
        data["probs"],
        models["pca"],
        models["gmm"],
    )


def predict_query_clusters(query_embedding: np.ndarray, pca: PCA, gmm: GaussianMixture) -> np.ndarray:
    reduced = pca.transform(query_embedding.reshape(1, -1))
    probs = gmm.predict_proba(reduced)
    return probs[0] 


def plot_cluster_2d(reduced: np.ndarray, probs: np.ndarray, metadatas: list[dict]):
    pca2 = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca2.fit_transform(reduced)
    dominant = np.argmax(probs, axis=1)
    entropy = -np.sum(probs * np.log(probs + 1e-9), axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sc = axes[0].scatter(coords[:, 0], coords[:, 1], c=dominant, cmap='tab20',
                         alpha=0.4, s=3)
    axes[0].set_title('Documents coloured by dominant cluster')
    plt.colorbar(sc, ax=axes[0])
    sc2 = axes[1].scatter(coords[:, 0], coords[:, 1], c=entropy, cmap='hot_r',
                          alpha=0.4, s=3)
    axes[1].set_title('Documents coloured by assignment entropy\n(yellow = uncertain boundary docs)')
    plt.colorbar(sc2, ax=axes[1])

    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), "../data/cluster_viz.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Cluster visualisation saved to {plot_path}")


def run_clustering_pipeline():
    store = get_store()
    ids, embeddings, metadatas = get_all_embeddings(store)
    print(f"Loaded {len(ids)} embeddings from ChromaDB.")
    
    
    reduced, pca = reduce_dimensions(embeddings)
    
    
    n_clusters = select_n_clusters(reduced)
    
    gmm = fit_gmm(reduced, n_clusters)
    probs = gmm.predict_proba(reduced) 
    dominant_clusters = np.argmax(probs, axis=1).tolist()
    analysis = analyse_clusters(gmm, probs, metadatas, ids)
    print("\n=== CLUSTER ANALYSIS ===")
    for k, profile in analysis["cluster_profiles"].items():
        print(f"\nCluster {k} ({profile['size']} docs, dominant: {profile['dominant_newsgroup']})")
        print(f"  Top newsgroups: {profile['top_newsgroups']}")

    print(f"\n=== BOUNDARY CASES ===")
    for doc in analysis["boundary_documents"][:5]:
        print(f"  [{doc['newsgroup']}] {doc['subject'][:60]} — entropy={doc['entropy']:.3f}")
        print(f"    Straddles: {[(c, round(p,3)) for c, p in doc['top_clusters']]}")
    save_cluster_results(ids, probs, pca, gmm)
    update_cluster_metadata(store, ids, dominant_clusters, probs.tolist())
    plot_cluster_2d(reduced, probs, metadatas)

    return ids, probs, pca, gmm, analysis


if __name__ == "__main__":
    run_clustering_pipeline()