"""Per-article workflow trace and standalone HTML report.

The trace lives in ``AgentState["workflow_report"]`` while the graph runs. It is
*not* written to the database. Each write produces three files in the issue
directory:

- ``workflow-report.html``: self-contained, article-oriented report
- ``workflow-report.json``: workflow-order event ledger
- ``workflow-report.meta.json``: tiny status file for the console

The HTML renderer groups the workflow-order event ledger by ``news_item_id`` so
each retrieved link gets one section containing its full chain.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment, FileSystemLoader

from app.config import settings
from app.utils.time import now_utc

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
_env = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    # The template filename ends in ".j2", so extension-based autoescape
    # detection would miss it. Force escaping for this diagnostic report.
    autoescape=True,
    trim_blocks=True,
    lstrip_blocks=True,
)

# A modest throttle keeps long select/draft loops from re-rendering a large
# HTML file on every progress checkpoint.
MIN_WRITE_INTERVAL_SECONDS = 3.0

STEP_ORDER = [
    "retrieved",
    "lookback_filter",
    "embedded",
    "cross_dedup",
    "article_fetch",
    "select",
    "select_cap",
    "intra_dedup_llm",
    "intra_dedup_emb",
    "topic_cap",
    "draft_input",
    "draft",
    "critique",
    "published",
]

STEP_LABELS = {
    "retrieved": "Retrieved",
    "lookback_filter": "Lookback window",
    "embedded": "Embedded",
    "cross_dedup": "Cross-issue dedup",
    "article_fetch": "Full article fetch",
    "select": "Quality filter",
    "select_cap": "Per-batch cap",
    "intra_dedup_llm": "Intra-issue dedup (AI)",
    "intra_dedup_emb": "Intra-issue dedup (embedding)",
    "topic_cap": "Per-topic cap",
    "draft_input": "Draft input",
    "draft": "Draft",
    "critique": "Critique",
    "published": "Publication",
}

STATUS_LABELS = {
    "info": "Info",
    "ok": "OK",
    "kept": "Kept",
    "dropped": "Dropped",
    "failed": "Failed",
    "cached": "Cached",
    "fetched": "Fetched",
    "fallback": "Fallback to summary",
    "skipped": "Skipped",
    "rewritten": "Rewritten",
    "partial": "Partial",
    "misconfigured": "Misconfigured",
    "not_reached": "Not reached",
}

# Steps whose dropped status ends an article's journey.
TERMINAL_DROP_STEPS = {
    "lookback_filter",
    "cross_dedup",
    "select",
    "select_cap",
    "intra_dedup_llm",
    "intra_dedup_emb",
    "topic_cap",
}


def report_enabled(state: Any) -> bool:
    if not isinstance(state, dict):
        return False
    ctx = state.get("workflow_report")
    return bool(isinstance(ctx, dict) and ctx.get("enabled"))


def init_report(
    state: dict[str, Any],
    *,
    enabled: bool,
    issue_id: str,
    slug: str,
    issue_date: Optional[str],
    config_name: str,
    language: str,
    lookback_start: Optional[str],
    lookback_end: Optional[str],
) -> None:
    """Seed the report context. Called once by runner.run_issue()."""
    state["workflow_report"] = {
        "enabled": bool(enabled),
        "issue_id": issue_id,
        "slug": slug,
        "issue_date": issue_date,
        "config_name": config_name,
        "language": language,
        "lookback_start": lookback_start,
        "lookback_end": lookback_end,
        "started_at": now_utc().isoformat(),
        "events": [],
        "articles": {},
        "globals": [],
        "last_write_at": 0.0,
        "_seq": 0,
        "_last_written_final": False,
    }


def _next_seq(ctx: dict[str, Any]) -> int:
    ctx["_seq"] = int(ctx.get("_seq", 0)) + 1
    return ctx["_seq"]


def _item_value(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def safe_url(value: Any) -> str:
    """Return only http(s) URLs so report anchors cannot execute javascript:."""
    text = str(value or "").strip()
    if text.startswith(("http://", "https://")):
        return text
    return ""


def _article_snapshot(item: Any) -> dict[str, Any]:
    return {
        "news_item_id": str(_item_value(item, "id", "")),
        "title": _item_value(item, "title") or "",
        "url": safe_url(_item_value(item, "url")),
        "source_domain": _item_value(item, "source_domain") or "",
        "summary": _item_value(item, "summary") or "",
        "published_at": _iso(_item_value(item, "published_at")),
        "ingested_via": _item_value(item, "ingested_via"),
        "source_feed_url": _item_value(item, "source_feed_url"),
        "source_query": _item_value(item, "source_query"),
        "topic_id": str(_item_value(item, "topic_id")) if _item_value(item, "topic_id") else None,
        "article_fetch_status": _item_value(item, "article_fetch_status"),
    }


def ensure_article(
    state: dict[str, Any],
    item: Any,
    *,
    topic_name: Optional[str] = None,
    via: Optional[str] = None,
    origin: Optional[str] = None,
    synthesized: bool = False,
) -> Optional[str]:
    """Create the article metadata row if absent. Returns the article id."""
    if not report_enabled(state):
        return None
    ctx = state["workflow_report"]
    snap = _article_snapshot(item)
    nid = snap["news_item_id"]
    if not nid:
        return None

    articles = ctx.setdefault("articles", {})
    existing = articles.get(nid)
    if existing is None:
        snap["topic_name"] = topic_name
        articles[nid] = snap
    elif topic_name and not existing.get("topic_name"):
        existing["topic_name"] = topic_name

    if synthesized and not any(
        e.get("news_item_id") == nid and e.get("step") == "retrieved"
        for e in ctx.get("events", [])
    ):
        record_event(
            state,
            news_item_id=nid,
            step="retrieved",
            status="info",
            topic_id=snap.get("topic_id"),
            topic_name=topic_name,
            reason="In this issue's lookback window; ingested before this run.",
            reason_source="system",
            data={
                "via": snap.get("ingested_via") or via,
                "origin": snap.get("source_feed_url") or snap.get("source_query") or origin,
                "synthesized": True,
                "title": snap.get("title"),
                "url": snap.get("url"),
                "source_domain": snap.get("source_domain"),
                "summary": snap.get("summary"),
                "published_at": snap.get("published_at"),
            },
        )
    return nid


def record_retrieved(
    state: dict[str, Any],
    item: Any,
    *,
    topic_name: Optional[str] = None,
    via: Optional[str] = None,
    origin: Optional[str] = None,
    content: Optional[str] = None,
) -> None:
    """Record the initial RSS/SearXNG input for a newly ingested item."""
    if not report_enabled(state):
        return
    try:
        nid = ensure_article(state, item, topic_name=topic_name, via=via, origin=origin)
    except Exception:
        logger.exception("workflow_report: record_retrieved failed")
        return
    if not nid:
        return
    snap = state["workflow_report"]["articles"].get(nid, {})
    record_event(
        state,
        news_item_id=nid,
        step="retrieved",
        status="info",
        topic_id=snap.get("topic_id"),
        topic_name=topic_name,
        reason=f"Discovered via {via or snap.get('ingested_via') or 'ingest'}.",
        reason_source="system",
        data={
            "via": via or snap.get("ingested_via"),
            "origin": origin or snap.get("source_feed_url") or snap.get("source_query"),
            "title": snap.get("title"),
            "url": snap.get("url"),
            "source_domain": snap.get("source_domain"),
            "summary": content if content is not None else snap.get("summary"),
            "published_at": snap.get("published_at"),
            "synthesized": False,
        },
    )


def _record_event(
    state: dict[str, Any],
    *,
    news_item_id: Optional[str],
    step: str,
    status: str,
    topic_id: Optional[str] = None,
    topic_name: Optional[str] = None,
    reason: Optional[str] = None,
    reason_source: Optional[str] = None,
    data: Optional[dict[str, Any]] = None,
) -> None:
    if not report_enabled(state) or not news_item_id:
        return
    ctx = state["workflow_report"]
    articles = ctx.setdefault("articles", {})
    article = articles.get(str(news_item_id))
    if article is None:
        article = {
            "news_item_id": str(news_item_id),
            "topic_id": str(topic_id) if topic_id else None,
            "topic_name": topic_name,
        }
        articles[str(news_item_id)] = article
    record = {
        "seq": _next_seq(ctx),
        "at": now_utc().isoformat(),
        "news_item_id": str(news_item_id),
        "topic_id": topic_id or article.get("topic_id"),
        "topic_name": topic_name or article.get("topic_name"),
        "step": step,
        "status": status,
        "reason": reason,
        "reason_source": reason_source,
        "data": data or {},
    }
    ctx.setdefault("events", []).append(record)
    if topic_name and article and not article.get("topic_name"):
        article["topic_name"] = topic_name


def record_event(
    state: dict[str, Any],
    *,
    news_item_id: Optional[str],
    step: str,
    status: str,
    topic_id: Optional[str] = None,
    topic_name: Optional[str] = None,
    reason: Optional[str] = None,
    reason_source: Optional[str] = None,
    data: Optional[dict[str, Any]] = None,
) -> None:
    """Best-effort wrapper: a reporting bug must never break the workflow."""
    try:
        _record_event(
            state,
            news_item_id=news_item_id,
            step=step,
            status=status,
            topic_id=topic_id,
            topic_name=topic_name,
            reason=reason,
            reason_source=reason_source,
            data=data,
        )
    except Exception:
        logger.exception("workflow_report: record_event failed")


def record_global(
    state: dict[str, Any],
    step: str,
    status: str = "info",
    *,
    reason: Optional[str] = None,
    data: Optional[dict[str, Any]] = None,
) -> None:
    """Append a run-level event (editorial, render, warnings, summaries)."""
    if not report_enabled(state):
        return
    try:
        ctx = state["workflow_report"]
        ctx.setdefault("globals", []).append(
            {
                "seq": _next_seq(ctx),
                "at": now_utc().isoformat(),
                "step": step,
                "status": status,
                "reason": reason,
                "data": data or {},
            }
        )
    except Exception:
        logger.exception("workflow_report: record_global failed")


def _outcome(article: dict[str, Any], events: list[dict[str, Any]]) -> dict[str, Any]:
    if any(
        e.get("step") == "published" and e.get("status") not in ("failed", "not_reached")
        for e in events
    ):
        return {"kind": "published", "label": "Published", "stage": "published", "reason": ""}
    failed_draft = [e for e in events if e.get("step") == "draft" and e.get("status") == "failed"]
    if failed_draft:
        last = failed_draft[-1]
        return {
            "kind": "draft_failed",
            "label": "Draft failed",
            "stage": "draft",
            "reason": last.get("reason") or "Draft call failed.",
        }
    drops = [e for e in events if e.get("step") in TERMINAL_DROP_STEPS and e.get("status") == "dropped"]
    if drops:
        last = drops[-1]
        return {
            "kind": "dropped",
            "label": f"Dropped at {STEP_LABELS.get(last.get('step'), last.get('step'))}",
            "stage": last.get("step"),
            "reason": last.get("reason") or "",
        }
    return {
        "kind": "not_processed",
        "label": "Not processed",
        "stage": None,
        "reason": "No terminal decision was recorded for this article.",
    }


def group_articles(ctx: dict[str, Any]) -> list[dict[str, Any]]:
    """Group workflow-order events by article, preserving first-seen order."""
    articles = ctx.get("articles") or {}
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for event in ctx.get("events") or []:
        nid = event.get("news_item_id")
        if not nid:
            continue
        if nid not in grouped:
            article = dict(articles.get(nid) or {})
            article.setdefault("news_item_id", nid)
            grouped[nid] = {"article": article, "steps": []}
            order.append(nid)
        grouped[nid]["steps"].append(event)

    out: list[dict[str, Any]] = []
    for nid in order:
        item = grouped[nid]
        steps = item["steps"]
        # Resolve the terminal outcome from real events before appending
        # display-only placeholders.
        item["outcome"] = _outcome(item["article"], steps)
        # Make the chain explicit: append placeholders for steps that were not
        # reached after the article's last recorded step.
        last_idx = -1
        for e in steps:
            try:
                last_idx = max(last_idx, STEP_ORDER.index(e.get("step")))
            except ValueError:
                continue
        if last_idx + 1 < len(STEP_ORDER):
            for missing in STEP_ORDER[last_idx + 1 :]:
                steps.append(
                    {
                        "seq": None,
                        "at": None,
                        "news_item_id": nid,
                        "topic_id": item["article"].get("topic_id"),
                        "topic_name": item["article"].get("topic_name"),
                        "step": missing,
                        "status": "not_reached",
                        "reason": None,
                        "reason_source": None,
                        "data": {},
                    }
                )
        item["index"] = len(out) + 1
        out.append(item)
    return out


def _issue_dir(ctx: dict[str, Any]) -> Path:
    return Path(settings.issues_dir) / str(ctx.get("slug") or "issue")


def _payload(ctx: dict[str, Any], grouped: list[dict[str, Any]], final: bool) -> dict[str, Any]:
    return {
        "version": 1,
        "issue": {
            "issue_id": ctx.get("issue_id"),
            "slug": ctx.get("slug"),
            "issue_date": ctx.get("issue_date"),
            "config_name": ctx.get("config_name"),
            "language": ctx.get("language"),
            "lookback_start": ctx.get("lookback_start"),
            "lookback_end": ctx.get("lookback_end"),
        },
        "run": {
            "started_at": ctx.get("started_at"),
            "finished_at": now_utc().isoformat() if final else None,
            "final": bool(final),
            "article_count": len(grouped),
        },
        "articles": grouped,
        "events": ctx.get("events") or [],
        "globals": ctx.get("globals") or [],
    }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _status_counts(grouped: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"published": 0, "dropped": 0, "draft_failed": 0, "not_processed": 0}
    for a in grouped:
        kind = (a.get("outcome") or {}).get("kind") or "not_processed"
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def write_report(state: dict[str, Any], *, final: bool = False) -> Optional[Path]:
    """Render all three artifacts. Best-effort; never raises into the graph."""
    if not report_enabled(state):
        return None
    ctx = state["workflow_report"]
    try:
        grouped = group_articles(ctx)
        payload = _payload(ctx, grouped, final)
        payload["warnings"] = state.get("warnings") or []
        payload["funnel"] = (state.get("meta") or {}).get("funnel") or {}
        html = _env.get_template("workflow_report.html.j2").render(
            issue=payload["issue"],
            run=payload["run"],
            articles=grouped,
            globals=ctx.get("globals") or [],
            counts=_status_counts(grouped),
            funnel=(state.get("meta") or {}).get("funnel") or {},
            warnings=(state.get("warnings") or []),
            step_labels=STEP_LABELS,
            status_labels=STATUS_LABELS,
            final=final,
        )
        out_dir = _issue_dir(ctx)
        html_path = out_dir / "workflow-report.html"
        _atomic_write(html_path, html)
        _atomic_write(
            out_dir / "workflow-report.json",
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        )
        meta = {
            "generated_at": now_utc().isoformat(),
            "final": bool(final),
            "status": "final" if final else "partial",
            "article_count": len(grouped),
            "counts": _status_counts(grouped),
            "slug": ctx.get("slug"),
        }
        _atomic_write(
            out_dir / "workflow-report.meta.json",
            json.dumps(meta, indent=2, ensure_ascii=False),
        )
        ctx["last_write_at"] = time.time()
        ctx["_last_written_final"] = bool(final)
        logger.info("workflow_report: wrote %s (%d articles)", html_path, len(grouped))
        return html_path
    except Exception:
        # A diagnostic report must never take down a generation run.
        logger.exception("workflow_report: write failed")
        return None


def write_if_due(state: dict[str, Any], *, final: bool = False) -> Optional[Path]:
    """Throttled live writer used by node/progress checkpoints."""
    if not report_enabled(state):
        return None
    ctx = state["workflow_report"]
    last = float(ctx.get("last_write_at") or 0.0)
    if not final and (time.time() - last) < MIN_WRITE_INTERVAL_SECONDS:
        return None
    return write_report(state, final=final)


def read_report_meta(slug: str) -> dict[str, Any]:
    """Read the tiny status file for the details endpoint. Never raises."""
    try:
        path = Path(settings.issues_dir) / slug / "workflow-report.meta.json"
        if not path.exists():
            return {}
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        logger.exception("workflow_report: could not read meta for %s", slug)
        return {}
