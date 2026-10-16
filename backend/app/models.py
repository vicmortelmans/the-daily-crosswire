import uuid
from datetime import date, datetime
from typing import Optional

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlmodel import Column, Field, Relationship, SQLModel

from app.config import settings
from app.utils.time import now_utc

EMBED_DIM = settings.embedding_dim


class NewspaperConfig(SQLModel, table=True):
    __tablename__ = "newspaper_configs"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str = Field(unique=True, index=True)
    schedule_kind: str = Field(default="interval")
    frequency_days: Optional[int] = Field(default=None)
    weekdays: Optional[list[int]] = Field(
        default=None,
        sa_column=Column(ARRAY(sa.Integer), nullable=True),
    )
    run_hour: int = Field(default=6)
    include_puzzles: bool = Field(default=True)
    active: bool = Field(default=True)
    max_stories_per_topic: Optional[int] = Field(default=None)
    # ISO 639-1 code: "en", "es", "fr", "it", "pt", "de". Drives draft +
    # editorial output language and template string localization.
    language: str = Field(default="en")
    # Email addresses that receive a PDF of the generated issue after
    # every successful render. Empty list disables delivery for this
    # config. Validated as EmailStr at the API boundary (schemas.py).
    newsletter_emails: list[str] = Field(
        default_factory=list,
        sa_column=Column(ARRAY(sa.Text), nullable=False, server_default="{}"),
    )
    created_at: datetime = Field(default_factory=now_utc, sa_type=sa.DateTime(timezone=False))
    updated_at: datetime = Field(default_factory=now_utc, sa_type=sa.DateTime(timezone=False))

    topics: list["Topic"] = Relationship(
        back_populates="config",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )
    issues: list["Issue"] = Relationship(
        back_populates="config",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class Topic(SQLModel, table=True):
    __tablename__ = "topics"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    config_id: uuid.UUID = Field(foreign_key="newspaper_configs.id", index=True)
    name: str
    preferred_sources: list[str] = Field(
        default_factory=list,
        sa_column=Column(ARRAY(sa.String), nullable=False, server_default="{}"),
    )
    search_hints: Optional[str] = Field(default=None)

    config: Optional[NewspaperConfig] = Relationship(back_populates="topics")
    feeds: list["RSSFeed"] = Relationship(
        back_populates="topic",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class RSSFeed(SQLModel, table=True):
    __tablename__ = "rss_feeds"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    topic_id: uuid.UUID = Field(foreign_key="topics.id", index=True)
    url: str
    last_polled_at: Optional[datetime] = Field(default=None, sa_type=sa.DateTime(timezone=False))
    last_etag: Optional[str] = Field(default=None)
    last_modified: Optional[str] = Field(default=None)

    topic: Optional[Topic] = Relationship(back_populates="feeds")


class NewsItem(SQLModel, table=True):
    __tablename__ = "news_items"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    url: str = Field(unique=True, index=True)
    normalized_title: str = Field(index=True)
    title: Optional[str] = Field(default=None)
    summary: Optional[str] = Field(default=None, sa_column=Column(sa.Text))
    source_domain: Optional[str] = Field(default=None, index=True)
    published_at: Optional[datetime] = Field(default=None, index=True, sa_type=sa.DateTime(timezone=False))
    topic_id: Optional[uuid.UUID] = Field(default=None, foreign_key="topics.id")
    # How this item was discovered. Populated at insert time by the
    # ingester. Legacy rows stay null; the drops UI renders no source
    # line for them.
    ingested_via: Optional[str] = Field(default=None, sa_column=Column(sa.String(length=16)))
    # RSS feed URL that produced this row (only when ingested_via='rss').
    source_feed_url: Optional[str] = Field(default=None, sa_column=Column(sa.Text))
    # SearXNG query text that produced this row (only when ingested_via='searxng').
    source_query: Optional[str] = Field(default=None, sa_column=Column(sa.Text))
    embedding: Optional[list[float]] = Field(
        default=None,
        sa_column=Column(Vector(EMBED_DIM), nullable=True),
    )
    fetched_at: datetime = Field(default_factory=now_utc, sa_type=sa.DateTime(timezone=False))


class Issue(SQLModel, table=True):
    __tablename__ = "issues"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    config_id: uuid.UUID = Field(foreign_key="newspaper_configs.id", index=True)
    slug: str = Field(unique=True, index=True)
    issue_date: date = Field(index=True)
    lookback_start: Optional[date] = Field(default=None)
    lookback_end: Optional[date] = Field(default=None)
    html_path: Optional[str] = Field(default=None)
    status: str = Field(default="pending", index=True)
    error: Optional[str] = Field(default=None, sa_column=Column(sa.Text))
    drafts_json: Optional[dict] = Field(default=None, sa_column=Column(JSONB, nullable=True))
    created_at: datetime = Field(default_factory=now_utc, sa_type=sa.DateTime(timezone=False))

    config: Optional[NewspaperConfig] = Relationship(back_populates="issues")
    story_log: list["StoryLog"] = Relationship(
        back_populates="issue",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"},
    )


class StoryLog(SQLModel, table=True):
    __tablename__ = "story_log"

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    issue_id: uuid.UUID = Field(foreign_key="issues.id", index=True)
    news_item_id: uuid.UUID = Field(foreign_key="news_items.id", index=True)
    topic_id: Optional[uuid.UUID] = Field(default=None, foreign_key="topics.id")
    rendered_title: str
    rendered_body: Optional[str] = Field(default=None, sa_column=Column(sa.Text))
    rendered_title_embedding: Optional[list[float]] = Field(
        default=None,
        sa_column=Column(Vector(EMBED_DIM), nullable=True),
    )
    created_at: datetime = Field(default_factory=now_utc, sa_type=sa.DateTime(timezone=False))

    issue: Optional[Issue] = Relationship(back_populates="story_log")


class IssueLog(SQLModel, table=True):
    """Persisted snapshot of the per-run log buffer. Written once at the
    end of run_issue() (finally block) so entries survive the backend
    restart that the in-memory buffer does not. Cascade-deleted when
    the owning issue is removed."""

    __tablename__ = "issue_logs"

    id: Optional[int] = Field(default=None, primary_key=True)
    issue_id: uuid.UUID = Field(
        foreign_key="issues.id",
        index=True,
        sa_column_kwargs={"nullable": False},
    )
    ts: datetime = Field(sa_column=Column(sa.DateTime(timezone=True), nullable=False))
    level: str = Field(sa_column=Column(sa.String(length=16), nullable=False))
    logger_name: str = Field(sa_column=Column(sa.String(length=128), nullable=False))
    msg: str = Field(sa_column=Column(sa.Text, nullable=False))
