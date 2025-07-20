import os
import numpy as np
import uuid
import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from app.data_models import (
    JobTitleEmbeddingMatchRequest,
    JobTitleEmbeddingMatch,

    JobTitleEmbeddingFuzzyMatchRequest,
    JobTitleEmbeddingFuzzyMatch,

    JobTitleCascadeMatchRequest,
    JobTitleCascadeMatch
)

load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")

DB_URL = os.getenv("DATABASE_URL")


def get_connection():
    return psycopg2.connect(DB_URL)

def _normalize(vec):
    return (vec / np.linalg.norm(vec)).tolist()


def insert_job_title(title: str, embedding: list[float]):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO job_titles (uuid, title, embedding)
        VALUES (%s, %s, %s)
        ON CONFLICT (title) DO NOTHING;
        """,
        (str(uuid.uuid4()), title, embedding),
    )
    conn.commit()
    cur.close()
    conn.close()


def job_title_embedding_match(request: JobTitleEmbeddingMatchRequest):
    
    # Embed the query
    query_embedding = model.encode(request.q)
    query_embedding = _normalize(query_embedding)

    conn = get_connection()
    cur = conn.cursor()
    # cur.execute(
    #     """
    #     SELECT uuid, title, score
    #     FROM (
    #         SELECT uuid, title, embedding <=> %s::vector AS score
    #         FROM job_titles
    #     ) AS scored
    #     ORDER BY score ASC
    #     LIMIT %s;
    #     """,
    #     (query_embedding, request.limit),
    # )
    cur.execute(
        """
        SELECT uuid, title, (1 - (embedding <=> %s::vector)) AS similarity
        FROM job_titles
        WHERE embedding <=> %s::vector < (1 - %s)
        ORDER BY similarity DESC
        LIMIT %s;
        """,
        (query_embedding, query_embedding, request.embedding_score_threshold, request.limit),
    )

    results = cur.fetchall()
    cur.close()
    conn.close()
    return [
        JobTitleEmbeddingMatch(
            id=row[0],
            name=row[1],
            # embedding_score=(1 - row[2])
            embedding_score=row[2]
        )
        for row in results
        # if (1 - row[2]) >= request.embedding_score_threshold
    ]

def job_title_embedding_fuzzy_match(
    request: JobTitleEmbeddingFuzzyMatchRequest
):
    
     # Embed the query
    query_embedding = model.encode(request.q)
    query_embedding = _normalize(query_embedding)

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        WITH scored AS (
            SELECT
                uuid,
                title,
                1 - (embedding <=> %s::vector) AS embedding_score,
                similarity(title, %s) AS fuzzy_score
            FROM job_titles
            WHERE
                (embedding <=> %s::vector) < %s
                OR title %% %s
        )
        SELECT
            uuid,
            title,
            embedding_score,
            fuzzy_score,
            (%s * embedding_score + %s * fuzzy_score) AS final_score
        FROM scored
        ORDER BY final_score DESC
        LIMIT %s;
        """,
        (
            query_embedding, request.q,
            query_embedding, request.embedding_score_threshold, request.q,
            request.embedding_score_weight, request.fuzzy_score_weight,
            request.limit,
        ),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        JobTitleEmbeddingFuzzyMatch(
            id=row[0],
            name=row[1],
            embedding_score=float(row[2]),
            fuzzy_score=float(row[3]),
            final_score=float(row[4]),
        )
        for row in rows
    ]

def job_title_cascade_match(
    request: JobTitleCascadeMatchRequest
):
    
     # Embed the query
    query_embedding = model.encode(request.q)
    query_embedding = _normalize(query_embedding)

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        WITH exact_match AS (
            SELECT uuid, title, 1.0 AS embedding_score, 1.0 AS fuzzy_score
            FROM job_titles
            WHERE LOWER(title) = LOWER(%s)
        ),
        fuzzy_candidates AS (
            SELECT uuid, title, similarity(title, %s) AS fuzzy_score
            FROM job_titles
            WHERE title %% %s
            ORDER BY fuzzy_score DESC
            LIMIT %s
        ),
        vector_ranked AS (
            SELECT
                fc.uuid,
                fc.title,
                1 - (jt.embedding <=> %s::vector) AS embedding_score,
                fc.fuzzy_score
            FROM fuzzy_candidates fc
            JOIN job_titles jt ON jt.uuid = fc.uuid
            ORDER BY embedding_score DESC
            LIMIT %s
        ),
        combined AS (
            SELECT * FROM exact_match
            UNION
            SELECT * FROM vector_ranked
        )
        SELECT
            uuid,
            title,
            embedding_score,
            fuzzy_score,
            (%s * embedding_score + %s * fuzzy_score) AS final_score
        FROM combined
        ORDER BY final_score DESC
        LIMIT %s;
        """,
        (
            request.q,  # exact match
            request.q, request.q, request.fuzzy_candidates,  # fuzzy
            query_embedding, request.fuzzy_candidates,  # embedding on fuzzy candidates
            request.embedding_score_weight, request.fuzzy_score_weight, request.limit
        ),
    )

    rows = cur.fetchall()
    cur.close()
    conn.close()

    return [
        JobTitleCascadeMatch(
            id=row[0],
            name=row[1],
            embedding_score=float(row[2]),
            fuzzy_score=float(row[3]),
            final_score=float(row[4]),
        )
        for row in rows
    ]
