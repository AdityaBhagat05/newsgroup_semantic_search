
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

ARCHIVE_PATH = os.path.join(os.path.dirname(__file__), "data/original/20_newsgroups.tar.gz")


def main():
    print("=" * 60)
    print("STEP 1/3 — Loading and cleaning corpus")
    print("=" * 60)
    from data_processor import load_corpus, get_newsgroup_stats
    docs = load_corpus(ARCHIVE_PATH, min_words=50)
    stats = get_newsgroup_stats(docs)
    print(f"\nPer-newsgroup document counts:")
    for ng, count in stats.items():
        print(f"  {ng:<40} {count}")

    print("\n" + "=" * 60)
    print("STEP 2/3 — Embedding and building ChromaDB")
    print("=" * 60)
    from vectordb import build_vector_db
    collection = build_vector_db(docs)

    print("\n" + "=" * 60)
    print("STEP 3/3 — Fuzzy clustering with GMM")
    print("=" * 60)
    from clustering import run_clustering_pipeline
    ids, probs, pca, gmm, analysis = run_clustering_pipeline()

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    print(f"  Documents embedded: {collection.count()}")
    print(f"  Clusters found: {gmm.n_components}")
    print(f"  High-uncertainty docs: {analysis['entropy_stats']['pct_high_uncertainty']:.1%}")
    print("\nStart the API server with:")
    print("  uvicorn src.main:app --host 0.0.0.0 --port 8000")
    print("\nOr with Docker:")
    print("  docker-compose up --build")


if __name__ == "__main__":
    main()