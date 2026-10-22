"""add article_text, article_fetched_at, and article_fetch_status to news_items

Revision ID: e1a2b3c4d5e6
Revises: d8a91c4e2b01
Create Date: 2026-10-21 21:30:00.000000

Adds:
- news_items.article_text: full body text extracted from the article's website
- news_items.article_fetched_at: timestamp when full text was fetched
- news_items.article_fetch_status: 'ok', 'fallback_summary', or 'failed'
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "e1a2b3c4d5e6"
down_revision: Union[str, None] = "d8a91c4e2b01"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "news_items",
        sa.Column("article_text", sa.Text(), nullable=True),
    )
    op.add_column(
        "news_items",
        sa.Column("article_fetched_at", sa.DateTime(timezone=False), nullable=True),
    )
    op.add_column(
        "news_items",
        sa.Column("article_fetch_status", sa.String(length=32), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("news_items", "article_fetch_status")
    op.drop_column("news_items", "article_fetched_at")
    op.drop_column("news_items", "article_text")

