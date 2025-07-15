"""add pg_trgm extension

Revision ID: 7569315c4503
Revises: 6223d13477e4
Create Date: 2025-07-15 07:39:12.633955

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from db.custom_operations import create_pgtrgm_extension, drop_pgtrgm_extension


# revision identifiers, used by Alembic.
revision: str = '7569315c4503'
down_revision: Union[str, Sequence[str], None] = '6223d13477e4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    create_pgtrgm_extension()


def downgrade() -> None:
    """Downgrade schema."""
    drop_pgtrgm_extension()
