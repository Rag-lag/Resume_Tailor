import uuid

import chromadb
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Each pipeline run indexes one resume and throws it away afterwards, so nothing
# needs to persist between runs. An in-memory client avoids leftover data on disk
# if a run crashes before cleanup.
client = chromadb.EphemeralClient()


def new_collection_name() -> str:
    """Unique per-run collection, so concurrent requests never see each other's chunks."""
    return f"resume_{uuid.uuid4().hex}"


def _get_collection(name: str):
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def index_resume_chunks(chunks: list[dict], collection_name: str) -> None:
    collection = _get_collection(collection_name)  # always fresh: one collection per run
    if not chunks:
        return

    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=True).tolist()
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{k: v for k, v in c.items() if k != "text"} for c in chunks]

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)


def delete_collection(collection_name: str) -> None:
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass  # already gone, or never created because the run failed early


def retrieve_relevant_chunks(queries: list[str], collection_name: str,
                             bullet_max_distance: float = 0.55,
                             skill_max_distance: float = 0.35) -> list[list[dict]]:
    """One result list per query. All queries are embedded and searched in a single batch."""
    if not queries:
        return []
    # Thresholds are per chunk type: bare skill names produce close-but-wrong
    # matches (C# -> C++ at ~0.46), so skills need a tighter cutoff than bullets.
    collection = _get_collection(collection_name)
    query_embeddings = embedder.encode(queries, normalize_embeddings=True).tolist()

    bullets = collection.query(
        query_embeddings=query_embeddings, n_results=3,
        where={"source_type": {"$in": ["experience", "project"]}},
        include=["documents", "metadatas", "distances"],
    )
    skills = collection.query(
        query_embeddings=query_embeddings, n_results=2,
        where={"source_type": "skill"},
        include=["documents", "metadatas", "distances"],
    )

    all_results = []
    for i in range(len(queries)):
        results = []
        for res, threshold in [(bullets, bullet_max_distance), (skills, skill_max_distance)]:
            for doc, meta, dist in zip(res["documents"][i], res["metadatas"][i], res["distances"][i]):
                if dist <= threshold:
                    results.append({"text": doc, "metadata": meta, "distance": dist})
        all_results.append(results)
    return all_results
