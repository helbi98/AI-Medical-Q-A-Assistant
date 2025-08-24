import os, json
from typing import List, Dict, Iterator
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

from nltk.tokenize import sent_tokenize


def load_json_docs(data_dir: str = "data") -> List[Dict]:
    """
    Load all JSON files from the data directory and return them as a list of dicts.
    Each record includes pmid, journal, and concatenated text (title + abstract).
    """
    docs = []
    for fn in os.listdir(data_dir):
        if fn.endswith(".json"):
            with open(os.path.join(data_dir, fn), encoding="utf-8") as f:
                rec = json.load(f)
            text = (rec.get("title", "") + "\n\n" + rec.get("abstract", "")).strip()
            if text:
                docs.append(
                    {
                        "pmid": rec.get("pmid"),
                        "journal": rec.get("journal"),
                        "text": text,
                    }
                )
    return docs


def iter_chunks(docs: List[Dict], max_words: int = 100, overlap_sentences: int = 1) -> Iterator[Dict]:
    """
    Sentence-aware chunking with overlap in sentences.
    Chunks are made by combining sentences up to max_words.
    Overlap is handled at the sentence level to avoid cutting mid-sentence.
    """
    for doc in docs:
        sentences = sent_tokenize(doc["text"])
        pmid = doc["pmid"]
        journal = doc["journal"]

        chunk_sentences = []
        word_count = 0
        chunk_id = 0

        for i, sent in enumerate(sentences):
            sent_words = len(sent.split())
            if word_count + sent_words > max_words:
                # Yield current chunk
                yield {
                    "text": " ".join(chunk_sentences).strip(),
                    "pmid": pmid,
                    "journal": journal,
                    "chunk": chunk_id,
                }
                # Prepare next chunk with overlap
                start_idx = max(0, len(chunk_sentences) - overlap_sentences)
                chunk_sentences = chunk_sentences[start_idx:] + [sent]
                word_count = sum(len(s.split()) for s in chunk_sentences)
                chunk_id += 1
            else:
                chunk_sentences.append(sent)
                word_count += sent_words

        # Yield leftover chunk
        if chunk_sentences:
            yield {
                "text": " ".join(chunk_sentences).strip(),
                "pmid": pmid,
                "journal": journal,
                "chunk": chunk_id,
            }