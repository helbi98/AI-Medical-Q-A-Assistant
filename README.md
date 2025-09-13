# MediGuru – AI Medical Q&A Assistant

**MediGuru** is an AI-powered medical question-answering system designed to provide fast, reliable, and referenced responses based on PubMed abstracts and clinical guidelines. It uses a **Retrieval-Augmented Generation (RAG)** pipeline with local LLMs and a persistent vector database for semantic search.

---

## Features

- End-to-end **RAG pipeline** for medical Q&A
- Preprocessing and **text-chunking** for scientific documents
- **Embeddings** built using SentenceTransformers
- **Persistent vector database** using ChromaDB for fast semantic search
- **Local open-source LLM integration** (Gemma-3B via Ollama) for grounded medical answers
- Streamlit-based **interactive web interface** with:
  - Synthesized answers
  - Source references
  - Safety disclaimers

---

## Architecture

1. **Data ingestion**
   - Load JSON files from PubMed or clinical guideline sources.
   - Each JSON contains `pmid`, `journal`, `title`, `abstract`.

2. **Preprocessing**
   - Concatenate title + abstract into a single text field.
   - Split text into **sentence-aware chunks**.

3. **Vector Store**
   - Each chunk is embedded separately using **SentenceTransformers**.
   - Embeddings are stored in a **persistent ChromaDB collection** for efficient retrieval.
   - Supports **incremental updates** without duplicating existing vectors.

4. **Retriever**
   - Queries are embedded with the same model used for chunks.
   - Top-k most relevant chunks retrieved via **cosine similarity** from ChromaDB.

5. **Generator (LLM)**
   - Local LLM (Gemma-3B) synthesizes answers based on retrieved chunks.
   - Produces grounded responses with references to original papers.

6. **Web Interface**
   - Implemented with **Streamlit**
   - Users can input questions and get responses with source citations.

---

## Install dependencies

pip install -r requirements.txt


## Start Ollama and Gemma3B

	- ollama serve

	### (Optional) Run Gemma3B locally if needed
	- ollama run gemma3:4b


## Build the vector index

python -m mediguru.embed_index


## Start the Streamlit app

streamlit run app.py


### Note: To add new files to data/ add your email-id to 'Entrez.email' in mediguru/fetch_pubmed.py


