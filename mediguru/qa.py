import os, pickle
from typing import List, Dict, Tuple
import faiss
from sentence_transformers import SentenceTransformer
from .llm_ollama import generate_answer

EMB_MODEL = os.environ.get(
    "MEDIGURU_EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


class LocalRAG:
    def __init__(self, index_dir: str = "index"):
        self.index_path = os.path.join(index_dir, "faiss.index")
        self.meta_path = os.path.join(index_dir, "meta.pkl")
        if not (os.path.exists(self.index_path) and os.path.exists(self.meta_path)):
            raise FileNotFoundError("FAISS index or metadata not found. Build the index first.")

        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            self.metas: List[Dict] = pickle.load(f)

        self.embedder = SentenceTransformer(EMB_MODEL)

    def search(self, query: str, k: int = 5) -> List[Dict]:
        q_emb = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        D, I = self.index.search(q_emb, k)
        return [self.metas[i] | {"score": float(D[0][j])} for j, i in enumerate(I[0])]

    def answer(self, query: str, model: str = "gemma3:1b", k: int = 5) -> Tuple[str, List[Dict]]:
        top_chunks = self.search(query, k=k)

        if "text" not in top_chunks[0]:
            raise RuntimeError("Metas do not include 'text'. Re-run indexing after updating embed_index.py.")

        answer = generate_answer(query, top_chunks, model=model)
        return answer, top_chunks
