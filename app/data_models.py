from typing import Optional
import uuid
from pydantic import BaseModel, Field, field_validator


class JobRoleSuggestionRequest(BaseModel):
    q: str
    limit: Optional[int] = Field(
        default=10, description="The upper limit of results to be returned"
    )
    threshold: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Similarity score threshold. Between 0 and 1.",
    )


class JobRoleSuggestion(BaseModel):
    id: uuid.UUID
    name: str
    score: float


class JobRoleSuggestionsResponse(BaseModel):
    suggestions: list[JobRoleSuggestion]


# ==== Ingestion Validation Model ====
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


class JobRole(BaseModel):
    uuid: uuid.UUID
    title: str
    code: Optional[str]
    job_zone: Optional[int]
    embedding: Optional[list[float]] = None
