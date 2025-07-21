# scripts/ingest_alt_titles.py
import csv
import numpy as np
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from app.job_title_match import insert_job_title

DATA_PATH = Path("raw_data/Alternate_Titles.csv")
EMBED_PARALLEL = True


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def load_alt_titles() -> List[tuple]:
    """Load Alternate Titles and ONET codes."""
    records = []
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            alt_title = row.get("Alternate Title")
            onet_code = row.get("O*NET-SOC Code")
            if alt_title and onet_code:
                records.append((alt_title.strip(), onet_code.strip()))
    return records


def embed_titles(records: List[tuple], model) -> List[tuple]:
    titles = [r[0] for r in records]
    embeddings = [None] * len(titles)  # pre-allocate results

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(model.encode, title): idx
            for idx, title in enumerate(titles)
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Embedding titles"
        ):
            idx = futures[future]
            embeddings[idx] = future.result()

    # Normalize embeddings
    embeddings = [_normalize(np.array(e)).tolist() for e in embeddings]
    return [
        (title, onet_code, emb) for (title, onet_code), emb in zip(records, embeddings)
    ]


def ingest():
    print("Loading alternate titles...")
    records = load_alt_titles()
    print(f"Loaded {len(records)} alternate titles.")

    print("Embedding...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    record_embeddings = embed_titles(records, model)

    print("Inserting into DB...")
    for title, onet_code, embedding in tqdm(
        record_embeddings, desc="Inserting alt titles"
    ):
        insert_job_title(
            title=title, embedding=embedding, type_="alt_title", onet_code=onet_code
        )


if __name__ == "__main__":
    ingest()
