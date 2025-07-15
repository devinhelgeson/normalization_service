from sqlmodel import SQLModel, Field
from typing import Optional
import uuid as uuid_pkg # need this because of field name uuid conflicts
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import Column, Index, Text
from pgvector.sqlalchemy import Vector

class JobTitle(SQLModel, table=True):
    __tablename__ = "job_titles"
    __table_args__ = (
        Index(
            "job_titles_title_trgm_idx",
            "title",
            postgresql_using="gin",
            postgresql_ops={"title": "gin_trgm_ops"}
        ),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    uuid: uuid_pkg.UUID = Field(sa_column=Column(UUID(as_uuid=True), nullable=False, unique=True))
    title: str = Field(sa_column=Column(Text, nullable=False, unique=True))
    embedding: Optional[list[float]] = Field(
        sa_column=Column(Vector(384), nullable=True)
    )