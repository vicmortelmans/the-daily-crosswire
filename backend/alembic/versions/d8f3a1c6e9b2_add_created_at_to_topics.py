"""add created_at to topics

Revision ID: d8f3a1c6e9b2
Revises: c4d9e2a7b1f3
Create Date: 2026-11-01 00:00:00.000000

Topics had no ordering column, so every query returned whatever order
PostgreSQL's heap happened to be in. An UPDATE writes a new row version at
the end of the heap, which made an edited topic jump to the bottom of the
config page. This adds a creation timestamp that UPDATE never changes.

Existing rows are backfilled in their current display order (heap order) so
the list does not visually jump when this migration runs.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "d8f3a1c6e9b2"
down_revision: Union[str, None] = "c4d9e2a7b1f3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("topics", sa.Column("created_at", sa.DateTime(), nullable=True))

    # Preserve the order the UI currently shows: number rows per config by
    # their physical position, then assign distinct timestamps. The last
    # existing topic lands one second in the past so a newly created topic
    # (which gets now_utc() from the ORM) always sorts after it.
    op.execute(
        """
        WITH ordered AS (
            SELECT
                id,
                row_number() OVER (PARTITION BY config_id ORDER BY ctid) AS rn,
                count(*) OVER (PARTITION BY config_id) AS total
            FROM topics
        )
        UPDATE topics AS t
        SET created_at = (now() AT TIME ZONE 'utc')
                         - ((o.total - o.rn + 1) * interval '1 second')
        FROM ordered AS o
        WHERE t.id = o.id
        """
    )

    op.alter_column("topics", "created_at", nullable=False)


def downgrade() -> None:
    op.drop_column("topics", "created_at")
