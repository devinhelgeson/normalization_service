from typing import List, Optional
import uuid
from pydantic import BaseModel, Field, field_validator

# JobTitleEmbeddingMatch

class JobTitleEmbeddingMatchRequest(BaseModel):
    q: str
    limit: Optional[int] = Field(
        default=10, description="The upper limit of results to be returned"
    )
    embedding_score_threshold: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Similarity score threshold. Between 0 and 1.",
    )

class JobTitleEmbeddingMatch(BaseModel):
    id: uuid.UUID
    name: str
    embedding_score: float

class JobTitleEmbeddingMatchResponse(BaseModel):
    matches: list[JobTitleEmbeddingMatch]

# JobTitleEmbeddingFuzzyMatch

class JobTitleEmbeddingFuzzyMatchRequest(JobTitleEmbeddingMatchRequest):
    embedding_score_weight: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Similarity score threshold. Between 0 and 1.",
    )
    fuzzy_score_weight: Optional[float] = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Similarity score threshold. Between 0 and 1.",
    )

class JobTitleEmbeddingFuzzyMatch(JobTitleEmbeddingMatch):
    fuzzy_score: float
    final_score: float

class JobTitleEmbeddingFuzzyMatchResponse(BaseModel):
    matches: List[JobTitleEmbeddingFuzzyMatch]

# JobTitleCascadeMatch

class JobTitleCascadeMatchRequest(JobTitleEmbeddingFuzzyMatchRequest):
    fuzzy_candidates: Optional[int] = Field(
        default=100, description="The number of initial fuzzy match search results"
    )

class JobTitleCascadeMatch(JobTitleEmbeddingFuzzyMatch):
    pass

class JobTitleCascadeMatchResponse(JobTitleEmbeddingFuzzyMatchResponse):
    matches: List[JobTitleCascadeMatch]

# Ingestion Validation
class RawOccupationRecord(BaseModel):
    code: str = Field(..., alias="Code")
    title: str = Field(..., alias="Occupation")
    job_zone: Optional[int] = Field(None, alias="Job Zone")
    data_level: Optional[str] = Field(None, alias="Data-level")

    @field_validator("job_zone", mode="before")
    @classmethod
    def clean_job_zone(cls, v):
        if isinstance(v, str) and v.strip().lower() == "n/a":
            return None
        return v

    def is_valid(self) -> bool:
        return self.data_level == "Y" and self.job_zone is not None


# class JobTitle(BaseModel):
#     uuid: uuid.UUID
#     title: str
#     code: Optional[str]
#     job_zone: Optional[int]
#     embedding: Optional[list[float]] = None