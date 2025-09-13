import os
import shutil
from typing import List, Dict
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from .preprocess import load_json_docs, iter_chunks

# ── Config ─────────────────────────────────────────────────────────────────────
EMB_MODEL = os.environ.get("MEDIGURU_EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DIR = "chroma_index"   # new directory for Chroma
COLLECTION_NAME = "mediguru"  # collection name inside Chroma


def build_chroma_index(data_dir: str = "data",
                       persist_dir: str = CHROMA_DIR,
                       collection_name: str = COLLECTION_NAME,
                       rebuild: bool = False) -> None:
    """
    Build or update a persistent ChromaDB collection from JSON docs in `data_dir`.

    - If rebuild=True, deletes and rebuilds the index from scratch.
    - If rebuild=False, performs incremental indexing:
        - Skips chunks that already exist (same deterministic ID).
        - Adds only new chunks from new/updated files.
    """
    # ── Rebuild or Incremental ────────────────────────────────────────────────
    if rebuild and os.path.exists(persist_dir):
        print(f"[Chroma] Removing existing directory: {persist_dir}")
        shutil.rmtree(persist_dir)

    os.makedirs(persist_dir, exist_ok=True)

    print(f"[Embeddings] Loading model: {EMB_MODEL}")
    embedding_model = SentenceTransformerEmbeddings(model_name=EMB_MODEL)

    # Load source docs
    docs = load_json_docs(data_dir)
    print(f"[Data] Loaded {len(docs)} papers from '{data_dir}'.")

    # Chunking
    chunk_stream = list(iter_chunks(docs, max_words=100, overlap_sentences=1))
    texts: List[str] = [c["text"] for c in chunk_stream]
    metadatas: List[Dict] = [
        {"pmid": c["pmid"], "journal": c["journal"], "chunk": c["chunk"]}
        for c in chunk_stream
    ]
    ids: List[str] = [f"{m['pmid']}:{m['chunk']}" for m in metadatas]

    if not texts:
        print("[Embeddings] No chunks to index. Exiting.")
        return

    # Open or create collection
    vectordb = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=persist_dir,
    )

    # ── Incremental Handling ──────────────────────────────────────────────────
    existing = vectordb.get(include=[])
    existing_ids = set(existing["ids"]) if existing and "ids" in existing else set()
    print(f"[Chroma] Existing vectors in DB: {len(existing_ids)}")

    # Select only new (non-duplicate) chunks
    new_texts, new_metas, new_ids = [], [], []
    for t, m, i in zip(texts, metadatas, ids):
        if i not in existing_ids:
            new_texts.append(t)
            new_metas.append(m)
            new_ids.append(i)

    print(f"[Chroma] New chunks to add: {len(new_ids)}")

    if new_texts:
        vectordb.add_texts(texts=new_texts, metadatas=new_metas, ids=new_ids)
        print(f"[Chroma] Added {len(new_ids)} new chunks.")
    else:
        print("[Chroma] No new chunks to add.")

    vectordb.persist()

    try:
        count = vectordb._collection.count()  # type: ignore[attr-defined]
        print(f"[Chroma] Persisted. Total vectors in collection: {count}")
    except Exception:
        print("[Chroma] Persisted.")


if __name__ == "__main__":
    build_chroma_index()
