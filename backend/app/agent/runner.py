import asyncio
import logging
import uuid
from datetime import date, timedelta

from sqlmodel import Session, select

from app.agent.graph import get_graph
from app.agent.state import AgentState
from app.config import settings
from app.db import engine
from app.models import Issue, NewspaperConfig, Topic
from app.services import log_buffer
from app.services import workflow_report
from app.utils.time import now_utc

logger = logging.getLogger(__name__)


def _slugify(s: str) -> str:
    out = []
    for ch in s.lower():
        if ch.isalnum() or ch in "-_":
            out.append(ch)
        elif ch == " ":
            out.append("-")
    slug = "".join(out).strip("-")
    return slug or "issue"


def _compute_lookback(config: NewspaperConfig, session: Session) -> tuple[date, date]:
    """Return (start, end) dates for this issue.

    If this config has a previous completed issue, start where that one left
    off. Otherwise fall back to a schedule-aware initial window:
      - interval mode -> frequency_days back
      - weekdays mode -> 7 days back (safe default for first-run)
    """
    today = date.today()
    last_issue = session.exec(
        select(Issue)
        .where(Issue.config_id == config.id)
        .where(Issue.status.in_(["drafted", "ready"]))
        .order_by(Issue.issue_date.desc())
        .limit(1)
    ).first()

    if last_issue is not None:
        start = last_issue.issue_date
    elif config.schedule_kind == "interval" and config.frequency_days:
        start = today - timedelta(days=max(config.frequency_days, 1))
    else:
        start = today - timedelta(days=7)

    return start, today


def create_pending_issue(config_id: uuid.UUID) -> uuid.UUID:
    """Create a fresh Issue row in 'pending' status for this config and
    return its id. Called synchronously by the API so the user can poll
    this exact issue while the scheduler picks up the job."""
    with Session(engine) as s:
        config = s.get(NewspaperConfig, config_id)
        if config is None:
            raise ValueError(f"config {config_id} not found")
        topics = s.exec(select(Topic).where(Topic.config_id == config.id)).all()
        if not topics:
            raise ValueError(f"config {config_id} has no topics")

        lookback_start, issue_date = _compute_lookback(config, s)
        slug_base = f"{issue_date.isoformat()}-{_slugify(config.name)}"
        slug = slug_base
        n = 1
        while s.exec(select(Issue).where(Issue.slug == slug)).first() is not None:
            n += 1
            slug = f"{slug_base}-{n}"

        issue = Issue(
            config_id=config.id,
            slug=slug,
            issue_date=issue_date,
            lookback_start=lookback_start,
            lookback_end=issue_date,
            status="pending",
        )
        s.add(issue)
        s.commit()
        s.refresh(issue)
        return issue.id


async def run_issue(issue_id: uuid.UUID) -> dict:
    """Run the agent graph on an already-created Issue. Called by both
    the scheduled path (via run_generation) and the manual path (via the
    generate endpoint's pre-created issue)."""
    with Session(engine) as s:
        issue = s.get(Issue, issue_id)
        if issue is None:
            raise ValueError(f"issue {issue_id} not found")
        config = s.get(NewspaperConfig, issue.config_id)
        if config is None:
            raise ValueError(f"config {issue.config_id} not found for issue {issue_id}")
        topics = s.exec(select(Topic).where(Topic.config_id == config.id)).all()
        issue.status = "running"
        s.add(issue)
        s.commit()

        issue_id_str = str(issue.id)
        config_id_str = str(config.id)
        topic_ids = [str(t.id) for t in topics]
        report_enabled = bool(config.include_workflow_report)
        report_ctx = {
            "slug": issue.slug,
            "issue_date": issue.issue_date.isoformat() if issue.issue_date else None,
            "lookback_start": (
                issue.lookback_start.isoformat() if issue.lookback_start else None
            ),
            "lookback_end": issue.lookback_end.isoformat() if issue.lookback_end else None,
            "config_name": config.name,
            "language": config.language,
        }

    logger.info(
        "agent run: issue=%s config=%s topics=%d",
        issue_id_str,
        config_id_str,
        len(topic_ids),
    )

    # Snapshot run identity up front so the UI can show which model +
    # tunables produced this issue without inferring from code/env state
    # at view time.
    chat_model = (
        settings.ollama_model
        if (settings.ollama_api_base and settings.ollama_model)
        else settings.openai_model
    )
    chat_endpoint = (
        settings.ollama_api_base
        if (settings.ollama_api_base and settings.ollama_model)
        else settings.openai_api_base
    )
    run_identity = {
        "started_at": now_utc().isoformat(),
        "chat_model": chat_model,
        "chat_endpoint": chat_endpoint,
        "embedding_model": settings.embedding_model,
        "select_batch_size": settings.select_batch_size,
        "select_max_kept_per_batch": settings.select_max_kept_per_batch,
        "select_intra_sim_threshold": settings.select_intra_sim_threshold,
        "dedup_sim_threshold": settings.dedup_sim_threshold,
        "llm_timeout_seconds": settings.llm_timeout_seconds,
        "llm_max_retries": settings.llm_max_retries,
    }

    initial: AgentState = {
        "issue_id": issue_id_str,
        "config_id": config_id_str,
        "topic_ids": topic_ids,
        "meta": {"run": run_identity},
    }
    workflow_report.init_report(
        initial,
        enabled=report_enabled,
        issue_id=issue_id_str,
        slug=report_ctx["slug"],
        issue_date=report_ctx["issue_date"],
        config_name=report_ctx["config_name"],
        language=report_ctx["language"],
        lookback_start=report_ctx["lookback_start"],
        lookback_end=report_ctx["lookback_end"],
    )

    # Route all agent log records into this issue's in-memory buffer so
    # the live dashboard Logs tab can tail them. The buffer is flushed
    # to the issue_logs DB table in the outer `finally` below so it
    # survives backend reloads + TTL eviction.
    log_buffer.start_buffer(issue_id_str)
    log_buffer.set_current_issue(issue_id_str)
    try:
        try:
            graph = get_graph()
            final_state: AgentState = await graph.ainvoke(initial)
            # The print node's timed wrapper usually writes the final report,
            # but keep this explicit so a custom/truncated graph still gets one.
            final_report = final_state.get("workflow_report")
            if (
                (not isinstance(final_report, dict) or not final_report.get("events"))
                and isinstance(initial.get("workflow_report"), dict)
                and initial["workflow_report"].get("events")
            ):
                final_state["workflow_report"] = initial["workflow_report"]
            workflow_report.write_report(final_state, final=True)
        except Exception as exc:
            logger.exception("agent graph failed")
            with Session(engine) as s:
                issue = s.get(Issue, issue_id)
                if issue is not None:
                    issue.status = "failed"
                    issue.error = str(exc)[:1000]
                    # Preserve the meta we've accumulated so the failed
                    # dashboard can show where the run died.
                    existing = dict(issue.drafts_json or {})
                    meta = dict(existing.get("meta") or {})
                    meta["failed_at_node"] = meta.get("current_node")
                    meta["error"] = str(exc)[:1000]
                    existing["meta"] = meta
                    issue.drafts_json = existing
                    s.commit()
            return {"issue_id": issue_id_str, "status": "failed", "error": str(exc)}
    finally:
        try:
            log_buffer.flush_to_db(issue_id_str)
        except Exception:
            logger.exception("log_buffer flush failed for issue %s", issue_id_str)
        log_buffer.set_current_issue(None)

    with Session(engine) as s:
        issue = s.get(Issue, issue_id)
        final_status = issue.status if issue else "unknown"
        html_path = issue.html_path if issue else None

    return {
        "issue_id": issue_id_str,
        "status": final_status,
        "html_path": html_path,
        "candidates": len(final_state.get("candidate_ids", [])),
        "kept": len(final_state.get("kept_ids", [])),
        "drafts": len(final_state.get("drafts", [])),
        "has_editorial": bool(final_state.get("editorial")),
        "warnings": final_state.get("warnings", []),
        "meta": final_state.get("meta", {}),
    }


async def run_generation(config_id: uuid.UUID) -> dict:
    """Scheduled entry point: create a pending Issue for the config, then
    run it."""
    issue_id = create_pending_issue(config_id)
    return await run_issue(issue_id)


def run_generation_sync(config_id_str: str) -> dict:
    """Synchronous entry point for APScheduler scheduled jobs (interval/cron).
    Accepts a config_id because APScheduler args are pickled."""
    cfg_id = uuid.UUID(config_id_str)
    try:
        return asyncio.run(run_generation(cfg_id))
    except Exception:
        logger.exception("run_generation_sync failed for config %s", cfg_id)
        raise


def run_issue_sync(issue_id_str: str) -> dict:
    """Synchronous entry point for APScheduler manual jobs. The endpoint
    creates the Issue up front so the response can include the id, then
    enqueues this function to actually run it."""
    issue_id = uuid.UUID(issue_id_str)
    try:
        return asyncio.run(run_issue(issue_id))
    except Exception:
        logger.exception("run_issue_sync failed for issue %s", issue_id)
        raise
