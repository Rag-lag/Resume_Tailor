from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer


DATA_DIR=Path("data")
CHROMA_DIR=Path(".chroma_db")
COLLECTION_NAME="raghav_profile"


# Don't need to do this cause ChromaDB applies it own embedding fn, but this for future proofing and control
# Bsically for now using same embed fn
# Ignore for now
class SBERTEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(self, model_name: str="all-MiniLM-L6-v2") -> None:
        self.model=SentenceTransformer(model_name)
    def __call__(self, input: Documents) -> Embeddings:
        embeddings=self.model.encode_document(list(input))
        return embeddings.tolist()

def get_collection():
    client=chromadb.PersistentClient(path=CHROMA_DIR)
    embedding_fn=SBERTEmbeddingFunction()
    # embedding_function=embedding_fn,
    return client.get_or_create_collection(name=COLLECTION_NAME,metadata={"description": "Candidate profile knowledge base"},)

def load_documents(data_dir: Path=DATA_DIR):
    docs=[]
    for file_name in sorted(os.listdir(data_dir)):
        if not file_name.endswith(".txt"):
            continue
        file_path=data_dir / file_name
        text=file_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        docs.append({
            "source": file_name,
            "content": text,
        })
    return docs

def infer_type_from_source(source: str):
    source_lower = source.lower()
    if "experience" in source_lower:
        return "experience"
    if "project" in source_lower:
        return "project"
    if "skill" in source_lower:
        return "skill"
    if "achievement" in source_lower:
        return "achievement"
    if "education" in source_lower:
        return "education"
    return "other"

def split_entries(text: str):
    parts = re.split(r"\n\s*---\s*\n", text)
    return [part.strip() for part in parts if part.strip()]

def extract_tags(entry: str):
    match = re.search(r"^Tags:\s*(.+)$", entry, flags=re.MULTILINE)
    if not match:
        return []

    raw_tags = match.group(1)
    return [tag.strip() for tag in raw_tags.split(",") if tag.strip()]

def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50):
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)

        if end == len(words):
            break

        start += max(1, chunk_size - overlap)

    return chunks

def prepare_chunks() -> Tuple[List[str], List[str], List[Dict[str, object]]]:
    documents = load_documents()
    ids: List[str] = []
    texts: List[str] = []
    metadatas: List[Dict[str, object]] = []

    for doc in documents:
        source = doc["source"]
        entry_type = infer_type_from_source(source)
        entries = split_entries(doc["content"])

        for entry_index, entry in enumerate(entries):
            tags = extract_tags(entry)
            chunks = chunk_text(entry)

            for chunk_index, chunk in enumerate(chunks):
                chunk_id = f"{source}::entry{entry_index}::chunk{chunk_index}"

                ids.append(chunk_id)
                texts.append(chunk)
                metadatas.append({
                    "source": source,
                    "type": entry_type,
                    "entry_index": entry_index,
                    "chunk_index": chunk_index,
                    "tags": ", ".join(tags),
                })

    return ids, texts, metadatas

def rebuild_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    existing_names = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_names:
        # print("Found old collection. Rebuilding it")
        client.delete_collection(COLLECTION_NAME)

    collection = get_collection()
    ids, texts, metadatas = prepare_chunks()

    if not ids:
        raise ValueError("No .txt data found in the data/ folder.")

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
    )
    # print("Ids:",ids)
    # print("Texts:",texts)
    # print("meta:",metadatas)
    
def retrieve(query: str, top_k: int = 6):
    collection = get_collection()

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )

    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0]

    retrieved: List[Dict[str, object]] = []
    for doc_id, doc_text, meta, distance in zip(ids, docs, metas, distances):
        retrieved.append({
            "id": doc_id,
            "text": doc_text,
            "metadata": meta,
            "distance": distance,
        })

    return retrieved

# rebuild_collection()
# r=retrieve("Looking for a software engineer who can build pipelins for a RAG based LLM Chatbot.")
# print("these are the relevant chunks:")
# for i in r:
#     print("text:",i['text'])
#     print("distance:",i['distance'])
    