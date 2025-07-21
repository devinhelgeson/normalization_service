from fastapi import APIRouter, Query
import numpy as np
from sentence_transformers import SentenceTransformer

from app.job_title_match import (
    job_title_embedding_match,
    job_title_embedding_fuzzy_match,
    job_title_cascade_match,
    job_title_kg_match,
)
from app.data_models import (
    JobTitleEmbeddingMatchRequest,
    JobTitleEmbeddingMatchResponse,
    JobTitleEmbeddingFuzzyMatchRequest,
    JobTitleEmbeddingFuzzyMatchResponse,
    JobTitleCascadeMatchRequest,
    JobTitleCascadeMatchResponse,
    JobTitleKGMatchRequest,
    JobTitleKGMatchResponse,
)

router = APIRouter()

# Initialize model once at module load
model = SentenceTransformer("all-MiniLM-L6-v2")


@router.get(
    "/match/embedding",
    response_model=JobTitleEmbeddingMatchResponse,
    summary="Get Normal Job Role Suggestions",
    description=(
        "For a given raw job title, this service performs semantic matching "
        "and outputs a normalized job role from the database."
    ),
    tags=["job"],
)
async def get_job_title_embedding_match(
    q: str = Query(..., description="Raw job title"),
    limit: int = Query(
        default=JobTitleEmbeddingMatchRequest.model_fields["limit"].default,
        description="Number of suggested job roles",
    ),
    embedding_score_threshold: float = Query(
        default=JobTitleEmbeddingMatchRequest.model_fields[
            "embedding_score_threshold"
        ].default,
        description="Minimum similarity score (0 to 1)",
    ),
):
    # Query DB for similar titles
    matches = job_title_embedding_match(
        JobTitleEmbeddingMatchRequest(
            q=q, limit=limit, embedding_score_threshold=embedding_score_threshold
        )
    )

    return JobTitleEmbeddingMatchResponse(matches=matches)


@router.get(
    "/match/embedding_fuzzy",
    summary="Get Job Role Suggestions (Embedding + Fuzzy Match)",
    description=(
        "Combines semantic similarity (pgvector) and fuzzy text match (pg_trgm) "
        "with configurable weights for scoring."
    ),
    tags=["job"],
)
async def get_job_title_embedding_fuzzy_match(
    q: str = Query(..., description="Raw job title"),
    limit: int = Query(
        default=JobTitleEmbeddingFuzzyMatchRequest.model_fields["limit"].default,
        description="Number of suggestions",
    ),
    embedding_score_threshold: float = Query(
        default=JobTitleEmbeddingMatchRequest.model_fields[
            "embedding_score_threshold"
        ].default,
        description="Minimum similarity score (0 to 1)",
    ),
    embedding_score_weight: float = Query(
        default=JobTitleEmbeddingFuzzyMatchRequest.model_fields[
            "embedding_score_weight"
        ].default,
        ge=0.0,
        le=1.0,
        description="Weight for embedding score",
    ),
    fuzzy_score_weight: float = Query(
        default=JobTitleEmbeddingFuzzyMatchRequest.model_fields[
            "fuzzy_score_weight"
        ].default,
        ge=0.0,
        le=1.0,
        description="Weight for fuzzy match score",
    ),
):
    # Normalize weights to sum to 1
    total = embedding_score_weight + fuzzy_score_weight
    if total == 0:
        embedding_score_weight, fuzzy_score_weight = 0.7, 0.3  # fallback
    else:
        embedding_score_weight /= total
        fuzzy_score_weight /= total

    # Get combined scores from DB
    matches = job_title_embedding_fuzzy_match(
        JobTitleEmbeddingFuzzyMatchRequest(
            q=q,
            limit=limit,
            embedding_score_threshold=embedding_score_threshold,
            embedding_score_weight=embedding_score_weight,
            fuzzy_score_weight=fuzzy_score_weight,
        )
    )

    # Return results in a structured response
    return JobTitleEmbeddingFuzzyMatchResponse(matches=matches)


@router.get(
    "/match/cascade",
    summary="Hybrid Search (Exact + Fuzzy + HNSW Embedding)",
    description="Efficient multi-step ranking using Postgres indexes for both fuzzy and embedding.",
    tags=["job"],
)
async def get_job_title_cascade_match(
    q: str = Query(...),
    limit: int = Query(
        default=JobTitleCascadeMatchRequest.model_fields["limit"].default,
        description="Number of suggestions",
    ),
    embedding_score_threshold: float = Query(
        default=JobTitleCascadeMatchRequest.model_fields[
            "embedding_score_threshold"
        ].default,
        description="Minimum similarity score (0 to 1)",
    ),
    embedding_score_weight: float = Query(
        default=JobTitleCascadeMatchRequest.model_fields[
            "embedding_score_weight"
        ].default,
        ge=0.0,
        le=1.0,
        description="Weight for embedding score",
    ),
    fuzzy_score_weight: float = Query(
        default=JobTitleCascadeMatchRequest.model_fields["fuzzy_score_weight"].default,
        ge=0.0,
        le=1.0,
        description="Weight for fuzzy match score",
    ),
    fuzzy_candidates: int = Query(
        default=JobTitleCascadeMatchRequest.model_fields["fuzzy_candidates"].default,
        description="Weight for fuzzy match score",
    ),
):
    # Normalize embedding
    query_embedding = model.encode(q)
    query_embedding = (query_embedding / np.linalg.norm(query_embedding)).tolist()

    # Run hybrid query
    matches = job_title_cascade_match(
        JobTitleCascadeMatchRequest(
            q=q,
            limit=limit,
            embedding_score_threshold=embedding_score_threshold,
            embedding_score_weight=embedding_score_weight,
            fuzzy_score_weight=fuzzy_score_weight,
            fuzzy_candidates=fuzzy_candidates,
        )
    )

    return JobTitleCascadeMatchResponse(matches=matches)


@router.get(
    "/match/kg",
    response_model=JobTitleKGMatchResponse,
    summary="Knowledge Graph Search",
    description="Search the vast collection of titles in the knowledge graph and hop across edges to return the proper ONET roles that might apply.",
    tags=["job"],
)
async def get_job_title_kg_match(
    q: str,
    limit: int = Query(
        default=JobTitleKGMatchRequest.model_fields["limit"].default,
        description="Number of suggestions",
    ),
    embedding_score_threshold: float = Query(
        default=JobTitleKGMatchRequest.model_fields[
            "embedding_score_threshold"
        ].default,
        description="Minimum similarity score (0 to 1)",
    ),
):
    matches = job_title_kg_match(
        JobTitleKGMatchRequest(
            q=q, limit=limit, embedding_score_threshold=embedding_score_threshold
        )
    )
    return matches
