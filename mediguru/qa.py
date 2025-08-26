import os
from typing import List, Dict, Tuple
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ── Config ─────────────────────────────────────────────────────────────────────
EMB_MODEL = os.environ.get("MEDIGURU_EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DIR = "chroma_index"
COLLECTION_NAME = "mediguru"


class LocalRAG:
    def __init__(self,
                 persist_dir: str = CHROMA_DIR,
                 collection_name: str = COLLECTION_NAME,
                 model_name: str = "gemma3:1b"):
        self.persist_dir = persist_dir

        # Basic existence check (folder with Chroma’s DB files)
        if not os.path.exists(self.persist_dir):
            raise FileNotFoundError(
                f"Chroma directory '{self.persist_dir}' not found. "
                f"Build it first by running embed_index.py."
            )

        # Embeddings + Vector DB
        self.embedding_model = SentenceTransformerEmbeddings(model_name=EMB_MODEL)
        self.vectordb = Chroma(
            collection_name=collection_name,
            embedding_function=self.embedding_model,
            persist_directory=self.persist_dir,
        )

        # Retriever
        self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 5})

        # LLM (Ollama)
        self.llm = Ollama(model=model_name)

        # PromptTemplate + LLMChain (answers are grounded in retrieved context)
        template = (
            "You are a careful medical research assistant. "
            "Answer ONLY using the provided context. "
            "Cite PMIDs where relevant. If the context is insufficient, say so "
            "and suggest what to search next.\n\n"
            "Question: {question}\n\n"
            "Context:\n{context}\n\n"
            "Answer:"
        )
        self.prompt = PromptTemplate(
            input_variables=["question", "context"],
            template=template,
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def _format_context(self, docs: List[Tuple]) -> str:
        """Make the context block readable and citable."""
        lines = []
        for i, (doc, _score) in enumerate(docs, start=1):
            pmid = doc.metadata.get("pmid", "?")
            journal = doc.metadata.get("journal", "?")
            chunk = doc.metadata.get("chunk", 0)
            snippet = doc.page_content.replace("\n", " ").strip()
            lines.append(f"[{i}] PMID:{pmid} | {journal} | chunk:{chunk}\n{snippet}")
        return "\n\n".join(lines)

    def answer(self, query: str, k: int = 5) -> Tuple[str, List[Dict]]:
        # Get docs with scores from Chroma
        docs_and_scores = self.vectordb.similarity_search_with_score(query, k=k)

        context = self._format_context(docs_and_scores)
        answer = self.chain.run(question=query, context=context).strip()

        # Assemble sources for UI
        sources: List[Dict] = []
        for doc, score in docs_and_scores:
            meta = doc.metadata or {}
            sources.append(
                {
                    "pmid": meta.get("pmid", "?"),
                    "journal": meta.get("journal", "?"),
                    "text": doc.page_content,
                    "score": float(score),
                }
            )

        return answer, sources
