import csv
import uuid
import numpy as np
from pathlib import Path
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer

from app.db import insert_job_title

from app.data_models import RawOccupationRecord

# ======= CONFIG =======
DATA_PATH = Path("raw_data/All_Occupations.csv")
EMBED_PARALLEL = True  # Set False to embed sequentially


# ======= LOAD & CLEAN =======
def load_valid_titles() -> List[RawOccupationRecord]:
    with open(DATA_PATH, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        records = []
        for row in reader:
            try:
                record = RawOccupationRecord(**row)
                if record.is_valid():
                    records.append(record)
                else:
                    print(f"[SKIP] Invalid or missing fields: {row}")
            except ValidationError as e:
                print(f"[ERROR] Could not parse row: {row} \nReason: {e}")
        return records


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)

def embed_titles(records: List[RawOccupationRecord], model) -> List[tuple]:
    titles = [r.title for r in records]
    if EMBED_PARALLEL:
        with ThreadPoolExecutor() as pool:
            embeddings = list(pool.map(model.encode, titles))
    else:
        embeddings = [model.encode(title) for title in titles]

    # Normalize all embeddings
    embeddings = [_normalize(np.array(e)).tolist() for e in embeddings]

    return list(zip(records, embeddings))



def ingest():
    print("Loading records...")
    records = load_valid_titles()
    print(f"Loaded {len(records)} valid job titles.")

    print("Embedding...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    record_embeddings = embed_titles(records, model)

    print("Inserting into DB...")
    for record, embedding in tqdm(record_embeddings, desc="Inserting job titles"):
        insert_job_title(title=record.title, embedding=embedding)

if __name__ == "__main__":
    ingest()
