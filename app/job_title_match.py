import os
from typing import Optional
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
    JobTitleCascadeMatch,
    JobTitleKGMatch,
    JobTitleKGMatchRequest,
    JobTitleKGMatchResponse,
    JobTitleKGMatchVariant,
)
from app.db_models import JobTitleType
from app.neo4j_client import neo4j_client

load_dotenv()

model = SentenceTransformer("all-MiniLM-L6-v2")

DB_URL = os.getenv("DATABASE_URL")


def get_connection():
    return psycopg2.connect(DB_URL)


def _normalize(vec):
    return (vec / np.linalg.norm(vec)).tolist()


def insert_job_title(
    title: str, embedding: list[float], type_: str, onet_code: Optional[str] = None
):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO job_titles (uuid, title, type, onet_code, embedding)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING;
        """,
        (str(uuid.uuid4()), title, type_, onet_code, embedding),
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
            AND type = 'onet_role'
        ORDER BY similarity DESC
        LIMIT %s;
        """,
        (
            query_embedding,
            query_embedding,
            request.embedding_score_threshold,
            request.limit,
        ),
    )

    results = cur.fetchall()
    cur.close()
    conn.close()
    return [
        JobTitleEmbeddingMatch(
            id=row[0],
            name=row[1],
            # embedding_score=(1 - row[2])
            embedding_score=row[2],
        )
        for row in results
        # if (1 - row[2]) >= request.embedding_score_threshold
    ]


def job_title_embedding_fuzzy_match(request: JobTitleEmbeddingFuzzyMatchRequest):
    # Embed the query
    query_embedding = model.encode(request.q)
    query_embedding = _normalize(query_embedding)

    conn = get_connection()
    cur = conn.cursor()
    # cur.execute(
    #     """
    #     WITH scored AS (
    #         SELECT
    #             uuid,
    #             title,
    #             1 - (embedding <=> %s::vector) AS embedding_score,
    #             similarity(title, %s) AS fuzzy_score
    #         FROM job_titles
    #         WHERE
    #             type = 'onet_role'
    #             AND (
    #                 (1 - (embedding <=> %s::vector)) >= %s
    #                 OR title %% %s
    #             )
    #     )
    #     SELECT
    #         uuid,
    #         title,
    #         embedding_score,
    #         fuzzy_score,
    #         (%s * embedding_score + %s * fuzzy_score) AS final_score
    #     FROM scored
    #     ORDER BY final_score DESC
    #     LIMIT %s;
    #     """,
    #     (
    #         query_embedding, request.q,
    #         query_embedding, request.embedding_score_threshold, request.q,
    #         request.embedding_score_weight, request.fuzzy_score_weight,
    #         request.limit,
    #     ),
    # )
    # rows = cur.fetchall()
    # cur.close()
    # conn.close()
    sql = """
        WITH scored AS (
            SELECT
                uuid,
                title,
                1 - (embedding <=> %s::vector) AS embedding_score,
                similarity(title, %s) AS fuzzy_score
            FROM job_titles
            WHERE type = 'onet_role'
                AND (title %% %s OR (1 - (embedding <=> %s::vector)) IS NOT NULL)
        ),
        filtered AS (
            SELECT *,
                (%s * embedding_score + %s * fuzzy_score) AS final_score
            FROM scored
        )
        SELECT
            uuid,
            title,
            embedding_score,
            fuzzy_score,
            final_score
        FROM filtered
        WHERE embedding_score >= %s
        ORDER BY final_score DESC
        LIMIT %s;
    """

    cur.execute(
        sql,
        (
            query_embedding,  # For embedding_score
            request.q,  # For fuzzy_score
            request.q,  # For title similarity
            query_embedding,  # For embedding presence
            request.embedding_score_weight,
            request.fuzzy_score_weight,
            request.embedding_score_threshold,  # Apply at the end
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


def job_title_cascade_match(request: JobTitleCascadeMatchRequest):
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
                AND type = 'onet_role'
        ),
        fuzzy_candidates AS (
            SELECT uuid, title, similarity(title, %s) AS fuzzy_score
            FROM job_titles
            WHERE title %% %s
                AND type = 'onet_role'
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
            request.q,
            request.q,
            request.fuzzy_candidates,  # fuzzy
            query_embedding,
            request.fuzzy_candidates,  # embedding on fuzzy candidates
            request.embedding_score_weight,
            request.fuzzy_score_weight,
            request.limit,
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


def job_title_kg_match(request: JobTitleKGMatchRequest):
    # 1. Embed the query
    query_embedding = model.encode(request.q)
    query_embedding = _normalize(query_embedding)

    # 2. Get top candidate from Postgres (embedding only, all titles)
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT uuid, title, type, onet_code, (1 - (embedding <=> %s::vector)) AS similarity
        FROM job_titles
        WHERE (1 - (embedding <=> %s::vector)) >= %s
        ORDER BY similarity DESC
        LIMIT %s;
        """,
        (
            query_embedding,
            query_embedding,
            request.embedding_score_threshold,
            request.limit,
        ),
    )

    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        return JobTitleKGMatchResponse(matches=[])

    # 3. Check the top result
    top_uuid, top_title, top_type, top_onet_code, top_score = rows[0]

    if top_type == JobTitleType.ONET_ROLE:
        # Top result is already a canonical ONET role → return as is
        return JobTitleKGMatchResponse(
            matches=[
                JobTitleKGMatch(
                    canonical_onet_title=top_title,
                    variants=[
                        JobTitleKGMatchVariant(
                            id=uuid.UUID(top_uuid),
                            name=top_title,
                            type=top_type,
                            embedding_score=top_score,
                        )
                    ],
                )
            ]
        )

    # 4. If top result is an alt_title → expand via KG
    query = """
    MATCH (alt:JobTitle {uuid: $uuid})-[:VARIANT_OF]->(onet:JobTitle)
    OPTIONAL MATCH (otherAlt:JobTitle)-[:VARIANT_OF]->(onet)
    RETURN onet.title AS canonical_title,
           collect({uuid: otherAlt.uuid, name: otherAlt.title, type: otherAlt.type}) AS variants
    """

    result = neo4j_client.run_query(query, {"uuid": str(top_uuid)})

    if not result:
        # If KG lookup fails, fallback to top match only
        return JobTitleKGMatchResponse(
            matches=[
                JobTitleKGMatch(
                    canonical_onet_title=top_title,
                    variants=[
                        JobTitleKGMatchVariant(
                            id=uuid.UUID(top_uuid),
                            name=top_title,
                            type=top_type,
                            embedding_score=top_score,
                        )
                    ],
                )
            ]
        )

    canonical_title = result[0]["canonical_title"]
    variants = [
        JobTitleKGMatchVariant(
            id=uuid.UUID(v["uuid"]),
            name=v["name"],
            type=v["type"],
            embedding_score=None,  # We don't score these; they come from KG
        )
        for v in result[0]["variants"]
    ]

    # Add the top alt_title with its score if it's not already in the variants
    if not any(str(top_uuid) == str(v.id) for v in variants):
        variants.append(
            JobTitleKGMatchVariant(
                id=uuid.UUID(top_uuid),
                name=top_title,
                type=top_type,
                embedding_score=top_score,
            )
        )

    return JobTitleKGMatchResponse(
        matches=[
            JobTitleKGMatch(canonical_onet_title=canonical_title, variants=variants)
        ]
    )
