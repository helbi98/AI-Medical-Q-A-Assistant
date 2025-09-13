import requests

OLLAMA_API_URL = "http://localhost:11434/api/generate"

SYSTEM_INSTRUCTION = (
    "You are a careful medical research assistant. "
    "Answer ONLY using the provided context. "
    "Cite PMIDs where relevant. If the context is insufficient, say so and suggest what to search."
)


def generate_answer(question: str, context_chunks: list, model: str = "gemma3:4b") -> str:
    pieces = []
    for c in context_chunks:
        meta = f"(PMID: {c.get('pmid','?')}, Journal: {c.get('journal','?')}, Chunk: {c.get('chunk',0)})"
        txt = c["text"].strip().replace("\n", " ")
        pieces.append(f"{meta}\n{txt}")

    # Truncate by chunks to avoid cutting mid-chunk
    max_chars = 8000
    context_pieces = []
    total_chars = 0
    for p in pieces:
        if total_chars + len(p) + 2 > max_chars:
            break
        context_pieces.append(p)
        total_chars += len(p) + 2 
    context = "\n\n".join(context_pieces)

    prompt = f"{SYSTEM_INSTRUCTION}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    payload = {"model": model, "prompt": prompt, "stream": False}

    response = requests.post(OLLAMA_API_URL, json=payload)
    response.raise_for_status()
    result = response.json()
    return result.get("response", "").strip()
