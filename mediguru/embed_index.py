import os, pickle
from typing import List, Dict
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from .preprocess import load_json_docs, iter_chunks

EMB_MODEL = os.environ.get(
    "MEDIGURU_EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

PROCESSED_FILE = "index/processed_pmids.txt"
INDEX_DIR = "index"
BATCH_SIZE = 32


def build_faiss_index(data_dir="data", index_dir=INDEX_DIR, batch_size=BATCH_SIZE):
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, "faiss.index")
    meta_path = os.path.join(index_dir, "meta.pkl")

    print(f"[Embeddings] Loading model: {EMB_MODEL}")
    model = SentenceTransformer(EMB_MODEL)

    # Load existing FAISS index if exists
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            metas: List[Dict] = pickle.load(f)
        processed_pmids = set([m["pmid"] for m in metas])
        print(f"[Index] Loaded existing index with {len(metas)} vectors.")
    else:
        index = None
        metas = []
        processed_pmids = set()
        print("[Index] Creating new FAISS index.")

    # Load JSON docs
    docs = load_json_docs(data_dir)
    new_docs = [doc for doc in docs if doc["pmid"] not in processed_pmids]
    print(f"[Embeddings] {len(new_docs)} new papers to embed.")

    # Convert to chunks
    chunk_stream = list(iter_chunks(new_docs, max_words=100, overlap_sentences=1))
    texts = [c["text"] for c in chunk_stream]

    if not texts:
        print("[Embeddings] No new chunks to embed. Exiting.")
        return

    print(f"[Embeddings] Encoding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Build or append FAISS index
    dim = embeddings.shape[1]
    if index is None:
        index = faiss.IndexFlatIP(dim)

    index.add(embeddings)

    # Update metas
    new_metas = [
        {
            "pmid": c["pmid"],
            "journal": c["journal"],
            "chunk": c["chunk"],
            "text": c["text"],
        }
        for c in chunk_stream
    ]
    metas.extend(new_metas)

    # Save FAISS index and meta
    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(metas, f)

    print(f"[Index] Wrote {index_path} and {meta_path} ({len(metas)} vectors).")


if __name__ == "__main__":
    build_faiss_index()
