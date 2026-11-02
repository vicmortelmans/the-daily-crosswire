"""add max_stories to topics

Revision ID: c4d9e2a7b1f3
Revises: f7c8d9e0a1b2
Create Date: 2026-11-01 00:00:00.000000

Adds a per-topic cap on how many stories one topic may contribute to a single
issue. Null means no topic-specific cap; the select node then falls back to
the config-wide newspaper_configs.max_stories_per_topic default (if set).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "c4d9e2a7b1f3"
down_revision: Union[str, None] = "f7c8d9e0a1b2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "topics",
        sa.Column("max_stories", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("topics", "max_stories")
