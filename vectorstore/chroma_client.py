import chromadb
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="resume_chunks",
    metadata={"hnsw:space": "cosine"}, 
)

def index_resume_chunks(chunks: list[dict]):
    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)

    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, normalize_embeddings=True).tolist() 
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{k: v for k, v in c.items() if k != "text"} for c in chunks]

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

def retrieve_relevant_chunks(query: str, n_results: int = 5,
                              bullet_max_distance: float = 0.55,
                              skill_max_distance: float = 0.35) -> list[dict]:
    query_embedding = embedder.encode([query], normalize_embeddings=True).tolist()

    bullets = collection.query(
        query_embeddings=query_embedding, n_results=3,
        where={"source_type": {"$in": ["experience", "project"]}},
        include=["documents", "metadatas", "distances"],
    )
    skills = collection.query(
        query_embeddings=query_embedding, n_results=2,
        where={"source_type": "skill"},
        include=["documents", "metadatas", "distances"],
    )

    results = []
    for res, threshold in [(bullets, bullet_max_distance), (skills, skill_max_distance)]:
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            if dist <= threshold:
                results.append({"text": doc, "metadata": meta, "distance": dist})
    return results