from Bio import Entrez
import json
import os
import time
from typing import List

Entrez.email = "****@gmail.com"

# Adjustable constants
DATA_DIR = "data"
RETMAX = 50          # number of papers per disease
SLEEP_TIME = 0.34    # 3 requests/sec limit

# Function to fetch PMIDs for a query
def fetch_pmids(query: str, retmax: int = RETMAX) -> List[str]:
    safe_query = query.replace("-", " ")
    handle = Entrez.esearch(db="pubmed", term=safe_query, retmax=retmax)
    rec = Entrez.read(handle)
    handle.close()
    return rec.get("IdList", [])

# Function to fetch a single record
def fetch_record(pmid: str):
    handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
    rec = Entrez.read(handle)
    handle.close()

    art = rec["PubmedArticle"][0]["MedlineCitation"]["Article"]
    title = art.get("ArticleTitle", "")
    abstract_list = art.get("Abstract", {}).get("AbstractText", [])
    abstract = "\n".join(abstract_list) if abstract_list else ""
    journal = art.get("Journal", {}).get("Title", "")

    return {"pmid": pmid, "title": title, "abstract": abstract, "journal": journal}

# Function to fetch and save all papers for a query
def fetch_and_save(query: str, out_dir: str = DATA_DIR, retmax: int = RETMAX):
    print(f"\n[Fetching] PubMed papers for: {query}")
    os.makedirs(out_dir, exist_ok=True)
    pmids = fetch_pmids(query, retmax=retmax)

    if not pmids:
        print(f"[Warning] No papers found for '{query}'. Skipping.")
        return

    added_count = 0
    skipped_count = 0

    for pmid in pmids:
        if not pmid.strip():
            continue

        path = os.path.join(out_dir, f"{pmid}.json")
        if os.path.exists(path):
            print(f"[Skipped] Already exists: {pmid}.json")
            skipped_count += 1
            continue

        try:
            rec = fetch_record(pmid)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)
            print(f"[Added] Saved new paper: {pmid}.json")
            added_count += 1
            time.sleep(SLEEP_TIME)
        except Exception as e:
            print(f"[Error] Failed to fetch PMID {pmid}: {e}")

    print(f"[Summary] Disease '{query}': {added_count} added, {skipped_count} skipped.")

# Function to fetch all diseases from a file
def fetch_all_diseases(disease_file: str = "diseases.txt"):
    if not os.path.exists(disease_file):
        print(f"[Error] Disease file '{disease_file}' not found.")
        return

    with open(disease_file, "r", encoding="utf-8") as f:
        diseases = [line.strip() for line in f if line.strip()]

    for disease in diseases:
        fetch_and_save(disease)

if __name__ == "__main__":
    fetch_all_diseases()
