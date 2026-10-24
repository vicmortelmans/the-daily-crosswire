import uuid
from typing import Any, Optional, TypedDict


class DraftDict(TypedDict, total=False):
    news_item_id: str
    topic_id: str
    topic_name: str
    rendered_title: str
    body: str
    is_lead: bool
    original_title: str
    original_url: str
    # Optional pull quote. `quote` is what the paper prints (translated into
    # the issue language); `quote_source` is the verified verbatim span from
    # the source article, kept for auditability in drafts_json / the report.
    quote: Optional[str]
    quote_source: Optional[str]
    quote_attribution: Optional[str]
    quote_verified: bool
    quote_translated: bool


class AgentState(TypedDict, total=False):
    # Inputs
    issue_id: str  # UUID as str (JSON serializable for langgraph)
    config_id: str
    topic_ids: list[str]

    # Per-node outputs
    candidate_ids: list[str]  # news_item ids after ingest/embed
    kept_ids: list[str]  # after dedupe
    selected_by_topic: dict[str, list[str]]  # topic_id -> [news_item_id, ...]
    lead_by_topic: dict[str, str]  # topic_id -> news_item_id
    quotes: dict[str, dict[str, Any]]  # news_item_id -> verified pull quote data
    drafts: list[DraftDict]
    editorial: Optional[str]

    # Puzzles
    puzzles_html: Optional[str]
    puzzles_meta: Optional[dict[str, Any]]

    # Bookkeeping
    warnings: list[str]
    meta: dict[str, Any]
    # In-memory workflow trace for the optional per-article HTML report.
    # Intentionally not copied into drafts_json / the database.
    workflow_report: dict[str, Any]
