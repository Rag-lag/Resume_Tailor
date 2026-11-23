import os
import tempfile

# huggingface_hub (used by fastembed to download) also writes to its own cache under
# HF_HOME, default ~/.cache, which is read-only on Vercel. Must be set before import.
os.environ.setdefault("HF_HOME", os.path.join(tempfile.gettempdir(), "huggingface"))

import numpy as np  # noqa: E402
from fastembed import TextEmbedding  # noqa: E402

# Same all-MiniLM-L6-v2 model as before, run through ONNX instead of PyTorch so the
# app fits Vercel's 500 MB Python bundle. Distances match the PyTorch version exactly,
# so the thresholds below still hold. Vercel only allows writes under /tmp.
_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2",
                       cache_dir=os.path.join(tempfile.gettempdir(), "fastembed"))


def _embed(texts: list[str]) -> np.ndarray:
    vectors = np.array(list(_model.embed(texts)), dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def build_index(chunks: list[dict]) -> dict:
    """One resume's chunks plus their vectors. Lives only in this run's state, so
    parallel requests can never see each other's data."""
    vectors = _embed([c["text"] for c in chunks]) if chunks else np.zeros((0, 384), np.float32)
    return {"chunks": chunks, "vectors": vectors}


def retrieve_relevant_chunks(queries: list[str], index: dict,
                             bullet_max_distance: float = 0.55,
                             skill_max_distance: float = 0.35) -> list[list[dict]]:
    """One result list per query: up to 3 bullets and 2 skills under their distance cutoff.
    Bullets and skills are ranked separately because short skill chunks otherwise crowd
    out richer bullets. Skills get a tighter cutoff: bare names give close-but-wrong
    matches (C# -> C++ at ~0.46)."""
    chunks = index["chunks"]
    if not queries:
        return []
    if not chunks:
        return [[] for _ in queries]

    distances = 1 - _embed(queries) @ index["vectors"].T  # cosine distance
    is_skill = np.array([c["source_type"] == "skill" for c in chunks])
    groups = [(np.flatnonzero(~is_skill), 3, bullet_max_distance),
              (np.flatnonzero(is_skill), 2, skill_max_distance)]

    all_results = []
    for row in distances:
        results = []
        for idx, top_k, threshold in groups:
            for i in idx[np.argsort(row[idx])[:top_k]]:
                if row[i] <= threshold:
                    meta = {k: v for k, v in chunks[i].items() if k != "text"}
                    results.append({"text": chunks[i]["text"], "metadata": meta, "distance": float(row[i])})
        all_results.append(results)
    return all_results
