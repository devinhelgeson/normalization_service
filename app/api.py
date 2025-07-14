from fastapi import APIRouter, Query
from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from app.db import query_similar_titles
from app.data_models import (
    JobRoleSuggestionRequest,
    JobRoleSuggestionsResponse,
    JobRoleSuggestion,
)

router = APIRouter()

# Initialize model once at module load
model = SentenceTransformer("all-MiniLM-L6-v2")


@router.get(
    "/suggestions",
    response_model=JobRoleSuggestionsResponse,
    summary="Get Normal Job Role Suggestions",
    description=(
        "For a given raw job title, this service performs semantic matching "
        "and outputs a normalized job role from the database."
    ),
    tags=["job"],
)
async def get_suggestions(
    q: str = Query(..., description="Raw job title"),
    limit: int = Query(
        default=JobRoleSuggestionRequest.model_fields["limit"].default,
        description="Number of suggested job roles",
    ),
    threshold: float = Query(
        default=JobRoleSuggestionRequest.model_fields["threshold"].default,
        description="Minimum similarity score (0 to 1)",
    ),
):
    # Embed the query
    query_embedding = model.encode(q)
    query_embedding = (query_embedding / np.linalg.norm(query_embedding)).tolist()


    # Query DB for similar titles
    results = query_similar_titles(query_embedding, top_k=limit)

    # Filter by threshold and format results
    suggestions = [
        JobRoleSuggestion(id=row["uuid"], name=row["title"], score = row["score"])
        for row in results
        # if (row["score"]) >= threshold
    ]

    return JobRoleSuggestionsResponse(suggestions=suggestions)