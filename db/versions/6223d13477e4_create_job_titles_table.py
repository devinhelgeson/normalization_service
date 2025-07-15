"""create job_titles table

Revision ID: 6223d13477e4
Revises: 
Create Date: 2025-07-15 06:49:20.337249

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = '6223d13477e4'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create job_titles table
    op.create_table(
        "job_titles",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("uuid", sa.dialects.postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(384), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("uuid"),
        sa.UniqueConstraint("title"),
    )

    # Create HNSW index for embeddings
    op.execute("""
        CREATE INDEX IF NOT EXISTS job_titles_embedding_hnsw_idx
        ON job_titles
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """)


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS job_titles_embedding_hnsw_idx")
    op.drop_table("job_titles")
    op.execute("DROP EXTENSION IF EXISTS vector")