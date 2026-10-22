import uuid
from datetime import date, datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from app.config import settings
from app.db import get_session
from app.models import Issue, NewspaperConfig, Topic
from app.schemas import (
    IssueDetail,
    IssueListItem,
    IssueSendRequest,
    IssueSendResponse,
    RecipientOutcome,
)
from app.services import log_buffer

router = APIRouter(prefix="/api/issues", tags=["issues"])


@router.get("", response_model=list[IssueListItem])
def list_issues(
    config_id: Optional[uuid.UUID] = None,
    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
    status_filter: Optional[str] = None,
    limit: int = 50,
    session: Session = Depends(get_session),
):
    query = select(Issue).order_by(Issue.issue_date.desc(), Issue.created_at.desc())
    if config_id:
        query = query.where(Issue.config_id == config_id)
    if from_date:
        query = query.where(Issue.issue_date >= from_date)
    if to_date:
        query = query.where(Issue.issue_date <= to_date)
    if status_filter:
        query = query.where(Issue.status == status_filter)
    query = query.limit(min(limit, 200))

    issues = session.exec(query).all()
    if not issues:
        return []

    # Batch the config-name lookup: one query for all configs referenced.
    config_ids = {i.config_id for i in issues}
    configs = session.exec(select(NewspaperConfig).where(NewspaperConfig.id.in_(config_ids))).all()
    name_by_id = {c.id: c.name for c in configs}

    return [
        {
            **IssueListItem.model_validate(i).model_dump(),
            "config_name": name_by_id.get(i.config_id),
        }
        for i in issues
    ]


@router.get("/{issue_id}", response_model=IssueDetail)
def get_issue(issue_id: uuid.UUID, session: Session = Depends(get_session)):
    issue = session.get(Issue, issue_id)
    if issue is None:
        raise HTTPException(status_code=404, detail="issue not found")
    cfg = session.get(NewspaperConfig, issue.config_id)

    drafts_json = issue.drafts_json or {}
    drafts = drafts_json.get("drafts", []) or []
    return {
        **IssueListItem.model_validate(issue).model_dump(),
        "config_name": cfg.name if cfg else None,
        "drafts_count": len(drafts),
        "has_editorial": bool(drafts_json.get("editorial")),
        "has_puzzles": bool(drafts_json.get("has_puzzles_html")),
        "warnings": drafts_json.get("warnings", []) or [],
    }


@router.get("/{issue_id}/details")
def get_issue_details(
    issue_id: uuid.UUID,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    """Rich metadata dump for the issue details dashboard.

    Older issues (generated before the drop_trace, funnel, and timings
    fields existed) return empty structures. The UI must handle missing
    fields gracefully rather than crash."""
    issue = session.get(Issue, issue_id)
    if issue is None:
        raise HTTPException(status_code=404, detail="issue not found")
    cfg = session.get(NewspaperConfig, issue.config_id)

    drafts_json = issue.drafts_json or {}
    drafts = drafts_json.get("drafts", []) or []
    meta: dict = drafts_json.get("meta") or {}
    warnings: list[str] = drafts_json.get("warnings") or []
    editorial_text = drafts_json.get("editorial")
    puzzles_meta = drafts_json.get("puzzles_meta") or {}

    topics = session.exec(select(Topic).where(Topic.config_id == issue.config_id)).all()
    topic_name_by_id = {str(t.id): t.name for t in topics}

    raw_drops = meta.get("drop_trace") or []
    drops: list[dict] = []
    for d in raw_drops:
        if not isinstance(d, dict):
            continue
        tid = d.get("topic_id")
        drops.append(
            {
                "news_item_id": d.get("news_item_id"),
                "stage": d.get("stage"),
                "topic_id": tid,
                "topic_name": topic_name_by_id.get(str(tid)) if tid else None,
                "reason": d.get("reason") or "",
                "snapshot": d.get("snapshot") or {},
            }
        )

    topic_counts = []
    select_meta = meta.get("select") or {}
    for tid, t in (select_meta.get("topics") or {}).items():
        topic_counts.append(
            {
                "topic_id": tid,
                "topic_name": t.get("topic_name") or topic_name_by_id.get(tid) or "",
                "candidates": t.get("candidates", 0),
                "batches": t.get("batches", 0),
                "kept": t.get("kept", 0),
                "collapsed_total": t.get("collapsed_total", 0),
                "has_lead": bool(t.get("has_lead")),
            }
        )
    topic_counts.sort(key=lambda r: (-r["kept"], r["topic_name"]))

    return {
        "issue_id": str(issue.id),
        "slug": issue.slug,
        "status": issue.status,
        "issue_date": issue.issue_date.isoformat() if issue.issue_date else None,
        "lookback_start": issue.lookback_start.isoformat() if issue.lookback_start else None,
        "lookback_end": issue.lookback_end.isoformat() if issue.lookback_end else None,
        "config_name": cfg.name if cfg else None,
        "drafts_count": len(drafts),
        "has_editorial": bool(editorial_text),
        "has_puzzles": bool(drafts_json.get("has_puzzles_html")),
        "warnings": warnings,
        "run": meta.get("run") or {},
        "timings": meta.get("timings") or {},
        "llm_calls": meta.get("llm_calls") or {},
        "ingest_rss": meta.get("ingest_rss") or {},
        "ingest_search": meta.get("ingest_search") or {},
        "embedding": meta.get("embedding") or {},
        "dedup_breakdown": meta.get("dedup_breakdown") or {},
        "select": {
            "total_kept": select_meta.get("total_kept", 0),
            "topics_with_keeps": select_meta.get("topics_with_keeps", 0),
            "by_topic": topic_counts,
        },
        "fetch_articles": meta.get("fetch_articles") or {},
        "draft": meta.get("draft") or {},
        "critique": meta.get("critique") or {},
        "editorial": meta.get("editorial") or {},
        "puzzles": {
            **(meta.get("puzzles") or {}),
            **({"generator_meta": puzzles_meta} if puzzles_meta else {}),
        },
        "render": meta.get("render") or {},
        "notify": meta.get("notify") or {},
        "print": meta.get("print") or {},
        "funnel": meta.get("funnel") or {},
        "drops": drops,
        "current_node": meta.get("current_node"),
        "completed_nodes": meta.get("completed_nodes") or [],
        "node_started_at": meta.get("node_started_at") or {},
        "failed_at_node": meta.get("failed_at_node"),
        "error": issue.error or meta.get("error"),
    }


@router.get("/{issue_id}/logs")
def get_issue_logs(
    issue_id: uuid.UUID,
    since: Optional[str] = None,
) -> dict[str, Any]:
    """Poll-based log tail for the dashboard Logs tab.

    Two-tier storage:
    - **In-memory ring buffer** (30-min TTL, 500-entry cap) for live and
      recently-finished runs. Fast; wiped on backend restart.
    - **issue_logs table** as the persistent record. flush_to_db() runs at
      every node boundary during a live run (plus select sub-steps), so
      older entries survive even if _evict_stale() wipes the ring during
      a quiet window.

    Returns the union of both sources, deduplicated by (ts, msg) and sorted
    by ts. Memory is freshest, DB has the durable tail. An issue with no
    logs in either place returns an empty list rather than a 404."""
    mem_entries = log_buffer.tail(str(issue_id), since=since)
    db_entries = log_buffer.load_from_db(str(issue_id), since=since)

    if mem_entries and db_entries:
        seen: set[tuple[str, str]] = set()
        merged: list[dict[str, Any]] = []
        # DB first; memory appended second overrides duplicates via the set.
        for e in db_entries:
            key = (e.get("ts", ""), e.get("msg", ""))
            if key not in seen:
                seen.add(key)
                merged.append(e)
        for e in mem_entries:
            key = (e.get("ts", ""), e.get("msg", ""))
            if key not in seen:
                seen.add(key)
                merged.append(e)
        merged.sort(key=lambda e: e.get("ts", ""))
        entries: list[dict[str, Any]] = merged
        source = "merged"
    elif mem_entries:
        entries = mem_entries
        source = "memory"
    elif db_entries:
        entries = db_entries
        source = "db"
    else:
        entries = []
        source = "empty"

    return {
        "issue_id": str(issue_id),
        "entries": entries,
        "count": len(entries),
        "source": source,
        # tz-aware datetime so the ISO string carries `+00:00`, matching
        # the shape of both in-memory buffer entries and DB-loaded entries.
        # Not using the project's `now_utc()` helper here because it strips
        # tzinfo for the naive-timestamp columns used elsewhere; this value
        # is a response field, not a DB write.
        "server_time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
    }


@router.post("/{issue_id}/send", response_model=IssueSendResponse)
async def send_issue(
    issue_id: uuid.UUID,
    payload: IssueSendRequest,
    session: Session = Depends(get_session),
) -> IssueSendResponse:
    """Ad-hoc delivery of an already-generated issue to a one-off recipient
    list. Distinct from the per-config newsletter which fires automatically
    via the notify node. Synchronous: blocks until SMTP returns so the UI
    can show per-recipient outcomes immediately."""
    issue = session.get(Issue, issue_id)
    if issue is None:
        raise HTTPException(status_code=404, detail="issue not found")
    if issue.status != "ready" or not issue.html_path:
        raise HTTPException(
            status_code=409,
            detail=f"issue is not ready (status={issue.status!r}); cannot send",
        )
    if not settings.smtp_configured:
        raise HTTPException(status_code=503, detail="SMTP not configured")

    # De-dupe while preserving order.
    seen: set[str] = set()
    recipients: list[str] = []
    for e in payload.emails:
        email = str(e).strip()
        if email and email not in seen:
            seen.add(email)
            recipients.append(email)
    if not recipients:
        raise HTTPException(status_code=400, detail="no valid recipients")

    config = session.get(NewspaperConfig, issue.config_id)
    paper_name = config.name if config else "The Daily Crosswire"
    language = config.language if config else "en"

    try:
        from app.services import mail as mail_service
        from app.services import pdf as pdf_service

        pdf_bytes = await pdf_service.build_issue_pdf(issue.slug)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"pdf generation failed: {type(exc).__name__}: {exc}",
        )

    payload_email = mail_service.build_issue_email(
        paper_name=paper_name,
        issue_date=issue.issue_date,
        language=language,
        slug=issue.slug,
        pdf_bytes=pdf_bytes,
        kind="adhoc",
    )

    try:
        outcomes = await mail_service.send_issue(recipients, payload_email)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"smtp transport failed: {type(exc).__name__}: {exc}",
        )

    sent = sum(1 for o in outcomes if o.get("status") == "ok")
    failed = sum(1 for o in outcomes if o.get("status") == "failed")
    return IssueSendResponse(
        issue_id=issue_id,
        sent=sent,
        failed=failed,
        per_recipient=[RecipientOutcome(**o) for o in outcomes],
    )
