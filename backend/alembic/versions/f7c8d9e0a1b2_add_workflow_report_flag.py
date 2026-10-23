"""add include_workflow_report to newspaper_configs

Revision ID: f7c8d9e0a1b2
Revises: e1a2b3c4d5e6
Create Date: 2026-10-23 00:00:00.000000

Adds a per-config switch that controls whether the agent writes a full
per-article workflow report into issues/{slug}/. The trace itself lives in
agent memory and on disk; no trace data is stored in Postgres.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "f7c8d9e0a1b2"
down_revision: Union[str, None] = "e1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "newspaper_configs",
        sa.Column(
            "include_workflow_report",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
    )


def downgrade() -> None:
    op.drop_column("newspaper_configs", "include_workflow_report")
