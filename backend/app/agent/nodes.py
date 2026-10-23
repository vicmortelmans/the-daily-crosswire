import asyncio
import logging
import os
from pathlib import Path
import tempfile
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import httpx
import openai
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from sqlalchemy import or_
from sqlmodel import Session, select

# Exceptions the LLM call sites should gracefully recover from. Anything
# else (TypeErrors, schema validation bugs, etc.) is a programmer error
# and should propagate so we see the traceback.
LLM_CATCHABLE = (
    openai.APIError,
    OutputParserException,
    httpx.HTTPError,
    TimeoutError,
    asyncio.TimeoutError,
)

from app.config import settings
from app.db import engine
from app.models import NewsItem, NewspaperConfig, RSSFeed, Topic
from app.services import dedup as dedup_service
from app.services import embeddings as emb_service
from app.services import log_buffer
from app.services import searxng as searxng_service
from app.services import workflow_report as report_service
from app.services.rss import normalize_title, poll_feed
from app.agent import prompts
from app.agent.checkpoint import checkpoint_meta
from app.agent.languages import hype_words, language_name
from app.agent.llm import (
    LLMUnavailable,
    chat_llm_configured,
    get_structured_llm,
)
from app.agent.state import AgentState, DraftDict
from app.puzzles.build import build_puzzles
from app.puzzles.nouns import extract_crypto_phrases, extract_theme_pool
from app.services import renderer as render_service
from app.utils.time import now_utc


def _load_issue_window(session: Session, state: AgentState) -> tuple[datetime, int]:
    """Return (lookback_start_datetime, window_days) based on the Issue row
    that runner.py created, NOT on config.frequency_days. This is the single
    source of truth for "what range of news counts for this run."""
    from app.models import Issue

    issue = session.get(Issue, _uuid(state["issue_id"]))
    if issue is None or issue.lookback_start is None:
        return now_utc() - timedelta(days=7), 7
    start_dt = datetime.combine(issue.lookback_start, datetime.min.time())
    end_date = issue.lookback_end or date.today()
    days = max((end_date - issue.lookback_start).days, 1)
    return start_dt, days


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers


def _uuid(s: str) -> uuid.UUID:
    return uuid.UUID(s)


def _warn(state: AgentState, msg: str) -> None:
    state.setdefault("warnings", []).append(msg)
    logger.warning("agent: %s", msg)


def _time_range_for_days(days: int) -> str:
    if days <= 1:
        return "day"
    if days <= 7:
        return "week"
    return "month"


def _record_llm_call(
    state: Optional[AgentState],
    stage: Optional[str],
    *,
    duration_ms: int,
    outcome: str,
    attempts_used: int,
) -> None:
    """Aggregate per-stage LLM call stats into meta.llm_calls. Each stage
    (select, draft, critique, editorial, intra_dedup) gets one rolling
    entry with call count, total duration, outcomes, and retry count.
    Skips recording if state/stage weren't passed in."""
    if state is None or not stage:
        return
    bucket = (
        state.setdefault("meta", {})
        .setdefault("llm_calls", {})
        .setdefault(
            stage,
            {
                "count": 0,
                "total_duration_ms": 0,
                "retries": 0,
                "timeouts": 0,
                "parse_errors": 0,
                "exhausted": 0,
            },
        )
    )
    bucket["count"] += 1
    bucket["total_duration_ms"] += duration_ms
    bucket["retries"] += max(0, attempts_used - 1)
    if outcome == "timeout":
        bucket["timeouts"] += 1
    elif outcome == "parse_error":
        bucket["parse_errors"] += 1
    elif outcome == "exhausted":
        bucket["exhausted"] += 1


async def _invoke_with_parse_retry(
    llm,
    messages,
    attempts: int = 3,
    *,
    state: Optional[AgentState] = None,
    stage: Optional[str] = None,
):
    """Call the LLM with retries on BOTH parse failures and timeouts.

    - Hard wall-clock cap per attempt via `asyncio.wait_for` using
      `settings.llm_timeout_seconds`. Catches silent hangs where the
      backend holds an open TCP connection but the remote never responds.
    - Up to `attempts` total calls (default 3). Retries on either
      `OutputParserException` (model produced non-conforming JSON) or
      `asyncio.TimeoutError` (hang). Mixing kinds across attempts is fine.
    - After all attempts fail, raise the LAST exception so the caller's
      `LLM_CATCHABLE` block can drop or fall through as configured.
    - When `state` and `stage` are passed, records aggregate call stats
      into `meta.llm_calls[stage]` for the UI to surface.

    Worst-case wall clock: attempts * llm_timeout_seconds. At defaults
    (3 * 300s) that's 15 min per failing call. Intentional: a model
    that is truly unresponsive deserves to be given up on, but not before
    real chances to recover.
    """
    last_exc: Optional[Exception] = None
    last_exc_kind: str = "exhausted"
    t0 = time.perf_counter()
    for attempt in range(1, attempts + 1):
        try:
            logger.info("DEBUG nodes.py AI messages %s", messages)
            result = await asyncio.wait_for(
                llm.ainvoke(messages),
                timeout=settings.llm_timeout_seconds,
            )
            logger.info("DEBUG nodes.py AI result %s", result)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            _record_llm_call(
                state,
                stage,
                duration_ms=elapsed_ms,
                outcome="ok",
                attempts_used=attempt,
            )
            return result
        except asyncio.TimeoutError as exc:
            last_exc = exc
            last_exc_kind = "timeout"
            logger.warning(
                "LLM call timed out after %ds (attempt %d/%d).",
                settings.llm_timeout_seconds,
                attempt,
                attempts,
            )
        except OutputParserException as exc:
            last_exc = exc
            last_exc_kind = "parse_error"
            logger.warning(
                "LLM output parse failed (attempt %d/%d): %s",
                attempt,
                attempts,
                str(exc)[:200],
            )
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    _record_llm_call(
        state,
        stage,
        duration_ms=elapsed_ms,
        outcome=last_exc_kind,
        attempts_used=attempts,
    )
    # Explicit check instead of `assert` (asserts get stripped when
    # Python is run with `-O`, which would turn this into an
    # `UnboundLocalError` on the `raise` below).
    if last_exc is None:
        raise RuntimeError(
            "LLM call exhausted retries but captured no exception; this " "should be unreachable"
        )
    raise last_exc


def _parse_searxng_published(raw: Optional[str]) -> Optional[datetime]:
    """SearXNG returns publishedDate as an ISO-8601 string when the upstream
    engine includes one. Parse into naive UTC to match our
    `timestamp without time zone` column. Returns None for missing or
    unparseable values. The caller treats None as 'undated, keep as
    fallback' in the lookback filter."""
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


# ---------------------------------------------------------------------------
# Node 1: ingest_rss: sync work inside asyncio thread


async def ingest_rss(state: AgentState) -> AgentState:
    def _work() -> dict:
        stats = {"feeds_polled": 0, "items_new": 0, "failures": 0}
        with Session(engine) as s:
            topics = s.exec(
                select(Topic).where(Topic.id.in_([_uuid(t) for t in state["topic_ids"]]))
            ).all()
            for topic in topics:
                feeds = s.exec(select(RSSFeed).where(RSSFeed.topic_id == topic.id)).all()
                for feed in feeds:
                    stats["feeds_polled"] += 1
                    try:
                        items = poll_feed(s, feed)
                        stats["items_new"] += len(items)
                        for item in items:
                            report_service.record_retrieved(
                                state,
                                item,
                                topic_name=topic.name,
                                via="rss",
                                origin=feed.url,
                                content=item.summary,
                            )
                        logger.info("agent ingest_rss: %s -> %d new", feed.url, len(items))
                    except Exception:
                        stats["failures"] += 1
                        logger.exception("ingest_rss failed for %s", feed.url)
        return stats

    stats = await asyncio.to_thread(_work)
    state.setdefault("meta", {})["ingest_rss"] = stats
    report_service.record_global(state, "ingest_rss", "ok" if not stats["failures"] else "failed", data=stats)
    return state


# ---------------------------------------------------------------------------
# Node 2: ingest_search: SearXNG for topics with no feeds or too-sparse RSS


async def ingest_search(state: AgentState, min_per_topic: int = 3) -> AgentState:
    stats = {"topics_queried": 0, "items_added": 0, "failures": 0}
    with Session(engine) as s:
        topics = s.exec(
            select(Topic).where(Topic.id.in_([_uuid(t) for t in state["topic_ids"]]))
        ).all()

        lookback_start, window_days = _load_issue_window(s, state)
        time_range = _time_range_for_days(window_days)

        for topic in topics:
            feeds = s.exec(select(RSSFeed).where(RSSFeed.topic_id == topic.id)).all()
            # Count recent items already tied to this topic. Same window
            # rule as embed: fetched + (published-in-window or undated).
            recent_count = len(
                s.exec(
                    select(NewsItem)
                    .where(NewsItem.topic_id == topic.id)
                    .where(NewsItem.fetched_at >= lookback_start)
                    .where(
                        or_(
                            NewsItem.published_at.is_(None),
                            NewsItem.published_at >= lookback_start,
                        )
                    )
                ).all()
            )
            if feeds and recent_count >= min_per_topic:
                continue

            query = topic.name
            if topic.preferred_sources:
                query += " (" + " OR ".join(f"site:{d}" for d in topic.preferred_sources[:3]) + ")"
            if topic.search_hints:
                query += f" {topic.search_hints}"

            stats["topics_queried"] += 1
            try:
                results = await searxng_service.search(query, time_range=time_range, max_results=15)
            except searxng_service.SearXNGError as exc:
                stats["failures"] += 1
                _warn(state, f"searxng failed for topic '{topic.name}': {exc}")
                continue

            # Bulk INSERT ... ON CONFLICT DO NOTHING so concurrent ingests
            # (e.g. a scheduled config's RSS poll) can't race us into a
            # UniqueViolation at commit time.
            from sqlalchemy.dialects.postgresql import insert as pg_insert

            rows = []
            seen_in_batch: set[str] = set()
            for r in results:
                url = r.get("url")
                title = r.get("title")
                if not url or not title or url in seen_in_batch:
                    continue
                seen_in_batch.add(url)
                rows.append(
                    {
                        "url": url,
                        "title": title,
                        "normalized_title": normalize_title(title),
                        "summary": (r.get("content") or "")[:2000] or None,
                        "source_domain": (url.split("/")[2] if "://" in url else None),
                        "published_at": _parse_searxng_published(r.get("published_date")),
                        "topic_id": topic.id,
                        "ingested_via": "searxng",
                        "source_query": query,
                    }
                )

            added = 0
            if rows:
                stmt = (
                    pg_insert(NewsItem)
                    .values(rows)
                    .on_conflict_do_nothing(index_elements=["url"])
                    .returning(NewsItem.id, NewsItem.url)
                )
                result_rows = s.execute(stmt).fetchall()
                added = len(result_rows)
                s.commit()
                row_by_url = {r["url"]: r for r in rows}
                for new_id, new_url in result_rows:
                    row = row_by_url.get(new_url) or {}
                    report_service.record_retrieved(
                        state,
                        {"id": new_id, **row},
                        topic_name=topic.name,
                        via="searxng",
                        origin=query,
                        content=row.get("summary"),
                    )
            stats["items_added"] += added
            logger.info(
                "agent ingest_search: topic=%r query=%r added=%d",
                topic.name,
                query,
                added,
            )
    state.setdefault("meta", {})["ingest_search"] = stats
    report_service.record_global(
        state, "ingest_search", "ok" if not stats["failures"] else "failed", data=stats
    )
    return state


# ---------------------------------------------------------------------------
# Node 3: embed: fill in embeddings for candidates in window


async def embed(state: AgentState) -> AgentState:
    with Session(engine) as s:
        lookback_start, _window_days = _load_issue_window(s, state)

        # Item counts as "in this run's window" iff:
        #   - we ingested it since lookback_start, AND
        #   - it was published since lookback_start OR has no publish date
        #     (we keep undated items as a safe fallback; some feeds omit it).
        candidates = s.exec(
            select(NewsItem)
            .where(NewsItem.topic_id.in_([_uuid(t) for t in state["topic_ids"]]))
            .where(NewsItem.fetched_at >= lookback_start)
            .where(
                or_(
                    NewsItem.published_at.is_(None),
                    NewsItem.published_at >= lookback_start,
                )
            )
        ).all()

        topics_by_id = {
            str(t.id): t.name
            for t in s.exec(
                select(Topic).where(Topic.id.in_([_uuid(t) for t in state["topic_ids"]]))
            ).all()
        }

        missing = [c for c in candidates if c.embedding is None]
        unembedded_count = 0
        newly_embedded_ids: set[str] = set()
        embedding_error: Optional[str] = None
        if missing:
            texts = [c.title or c.normalized_title or "" for c in missing]
            try:
                vectors = await emb_service.embed(texts)
                logger.info(f"DEBUG nodes.py successfully embedded {len(missing)} texts")
            except Exception as exc:
                _warn(
                    state,
                    f"embedding failed for {len(missing)} candidates ({exc}). "
                    "Proceeding without vectors; semantic dedup will be skipped for them.",
                )
                vectors = []
                embedding_error = f"{type(exc).__name__}: {exc}"

            if vectors and len(vectors) == len(missing):
                for item, vec in zip(missing, vectors):
                    item.embedding = vec
                    newly_embedded_ids.add(str(item.id))
                s.commit()
            else:
                unembedded_count = len(missing)

        # Establish the article list for this run. Items ingested in earlier
        # runs still enter the window, so they get a synthesized retrieval line.
        for candidate in candidates:
            report_service.ensure_article(
                state,
                candidate,
                topic_name=topics_by_id.get(str(candidate.topic_id)),
                synthesized=True,
            )

        # Any link retrieved during this run that did not become a candidate
        # was excluded by the lookback window. Record that decision explicitly.
        candidate_ids = {str(c.id) for c in candidates}
        ctx = state.get("workflow_report") or {}
        if report_service.report_enabled(state):
            for nid, article in (ctx.get("articles") or {}).items():
                if nid in candidate_ids:
                    continue
                report_service.record_event(
                    state,
                    news_item_id=nid,
                    step="lookback_filter",
                    status="dropped",
                    topic_id=article.get("topic_id"),
                    topic_name=article.get("topic_name"),
                    reason="Outside this issue's lookback window.",
                    reason_source="system",
                    data={"lookback_start": lookback_start.isoformat()},
                )

        for candidate in candidates:
            nid = str(candidate.id)
            if nid in newly_embedded_ids:
                status = "ok"
                note = None
            elif candidate.embedding is not None:
                status = "ok"
                note = "already embedded"
            else:
                status = "failed"
                note = embedding_error or "no embedding returned"
            report_service.record_event(
                state,
                news_item_id=nid,
                step="embedded",
                status=status,
                topic_id=str(candidate.topic_id) if candidate.topic_id else None,
                topic_name=topics_by_id.get(str(candidate.topic_id)),
                reason=note,
                reason_source="system",
                data={
                    "model": settings.embedding_model,
                    "already_embedded": nid not in newly_embedded_ids,
                    "error": embedding_error if status == "failed" else None,
                },
            )

        state.setdefault("meta", {})["embedding"] = {
            "total_candidates": len(candidates),
            "newly_embedded": 0 if unembedded_count else len(missing),
            "unembedded": unembedded_count,
        }

        state["candidate_ids"] = [str(c.id) for c in candidates]
        logger.info("agent embed: %d candidates, %d embedded now", len(candidates), len(missing))
    return state


# ---------------------------------------------------------------------------
# Node 4: dedupe


async def dedupe(state: AgentState) -> AgentState:
    with Session(engine) as s:
        # Dedup window is intentionally longer than the ingest window so a
        # weekly paper doesn't re-run the same story next week. It's derived
        # from the effective window this run uses, not from config directly.
        _start, window_days = _load_issue_window(s, state)
        window = dedup_service.dedup_window_days(window_days)
        candidates = s.exec(
            select(NewsItem).where(
                NewsItem.id.in_([_uuid(i) for i in state.get("candidate_ids", [])])
            )
        ).all()
        kept, decisions = dedup_service.filter_fresh(s, candidates, window_days=window)
        state["kept_ids"] = [str(c.id) for c in kept]
        state.setdefault("meta", {})["dedup_breakdown"] = {
            "total": len(candidates),
            "kept": len(kept),
            "dropped_url": sum(1 for d in decisions if d.decision == "dropped_url"),
            "dropped_title": sum(1 for d in decisions if d.decision == "dropped_title"),
            "dropped_similarity": sum(1 for d in decisions if d.decision == "dropped_similarity"),
        }
        cand_by_id = {str(c.id): c for c in candidates}
        for d in decisions:
            nid = str(d.candidate_id)
            n = cand_by_id.get(nid)
            if n is not None:
                report_service.ensure_article(state, n, synthesized=True)
            if d.decision == "kept":
                report_service.record_event(
                    state,
                    news_item_id=nid,
                    step="cross_dedup",
                    status="kept",
                    topic_id=str(n.topic_id) if n and n.topic_id else None,
                    reason="Not present in the recent story_log window.",
                    reason_source="system",
                )
                continue
            info = d.match_info or {}
            if d.decision == "dropped_url":
                reason = "url already in story_log window"
            elif d.decision == "dropped_title":
                reason = "normalized title already in story_log window"
            else:
                sim = info.get("similarity")
                matched = info.get("matched_title") or ""
                reason = (
                    f"similar to recent story (sim={sim:.3f}): {matched[:60]}"
                    if isinstance(sim, (int, float))
                    else "similar to recent story in story_log window"
                )
            _record_drop(
                state,
                news_item_id=nid,
                stage=d.decision,
                topic_id=str(n.topic_id) if n and n.topic_id else None,
                reason=reason,
                snapshot=_snapshot_news_item(n) if n else None,
            )
            report_service.record_event(
                state,
                news_item_id=nid,
                step="cross_dedup",
                status="dropped",
                topic_id=str(n.topic_id) if n and n.topic_id else None,
                reason=reason,
                reason_source="system",
                data={
                    "decision": d.decision,
                    "similarity": info.get("similarity"),
                    "matched_title": info.get("matched_title"),
                },
            )
        logger.info("agent dedupe: %d candidates -> %d kept", len(candidates), len(kept))
    return state


# ---------------------------------------------------------------------------
# Node 5: select: LLM quality filter


class _SelectOutput(BaseModel):
    """Flat schema kept deliberately simple. List-of-objects schemas cause
    smaller models (Gemma 4 26B MoE and similar) to stall or loop during
    generation. Two flat lists and one flat dict of strings land in ~60
    output tokens and keep the structured call fast and reliable on qwen3:14b."""

    keep_ids: list[str] = Field(
        description="news_item_id values to keep. Drop junk by leaving its id out."
    )
    lead_id: Optional[str] = Field(
        default=None,
        description="One id from keep_ids to feature as the lead story, or null if none.",
    )
    drop_reasons: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Map of news_item_id -> short one-line reason (under 80 chars) "
            "for every candidate NOT in keep_ids. Factual and mechanical "
            "(e.g. 'trailer-only, no near-term release', 'merch/promo card')."
        ),
    )
    keep_reasons: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional map of news_item_id -> short one-line reason for why "
            "each kept candidate was selected. Used by the workflow report."
        ),
    )


def _format_candidate_line(c: NewsItem) -> str:
    # Use the article body when available; otherwise fall back to the
    # summary. The RSS/search teaser is too shallow for quality selection.
    article_text = (c.article_text or "").strip()
    summary_text = (c.summary or "").strip()
    context_text = article_text[:2500] if article_text else summary_text[:250]
    return (
        f"- id={c.id}  title={c.title!r}  source={c.source_domain!r}"
        f"  article_text={context_text!r}"
    )


async def _select_chunk(
    llm,
    topic_name: str,
    chunk: list[NewsItem],
    state: Optional[AgentState] = None,
) -> _SelectOutput:
    """Run the `select` LLM call on one chunk of candidates. Raises
    LLM_CATCHABLE on failure; caller decides how to recover per-chunk."""
    prompt_lines = [_format_candidate_line(c) for c in chunk]
    user_msg = (
        prompts.SELECT_INSTRUCTION.format(topic_name=topic_name)
        + "\n\nUse the article_text field when available; it is the full body of the article."
        + "\n\nCandidates:\n"
        + "\n".join(prompt_lines)
    )
    return await _invoke_with_parse_retry(
        llm,
        [SystemMessage(content=prompts.SYSTEM_PROMPT), HumanMessage(content=user_msg)],
        state=state,
        stage="select",
    )


def _snapshot_news_item(n: NewsItem) -> dict:
    """Denormalized copy of the fields we want to show in a drop trace. Kept
    inline in drafts_json.meta so drops survive even if the source
    news_items row gets garbage-collected later.

    `ingested_via` + `source_feed_url` / `source_query` drive the drop
    card's "via RSS:..." / "via SearXNG:..." line. Legacy news_items rows
    (pre-source-tracking) have these as None; the UI renders nothing for
    those cases."""
    return {
        "title": (n.title or "")[:200],
        "url": n.url or "",
        "source_domain": n.source_domain or "",
        "summary": (n.summary or "")[:300],
        "published_at": n.published_at.isoformat() if n.published_at else None,
        "ingested_via": n.ingested_via,
        "source_feed_url": n.source_feed_url,
        "source_query": n.source_query,
    }


def _record_drop(
    state: AgentState,
    *,
    news_item_id: str,
    stage: str,
    topic_id: Optional[str],
    reason: str,
    snapshot: Optional[dict],
) -> None:
    """Append one entry to meta.drop_trace. The UI consumes this to show
    'why wasn't X in the paper?' without needing the live news_items row."""
    trace = state.setdefault("meta", {}).setdefault("drop_trace", [])
    trace.append(
        {
            "news_item_id": news_item_id,
            "stage": stage,
            "topic_id": topic_id,
            "reason": reason,
            "snapshot": snapshot,
        }
    )


def _content_text_for_dedup(title: Optional[str], summary: Optional[str]) -> str:
    """Build the text we embed for intra-issue dedup. Title-only embeddings
    cluster poorly on short Spanish news-outlet paraphrases (same event
    across outlets lands 0.60-0.84 cosine; junk landing pages cluster at
    0.85-1.00). Title + summary adds enough specificity to separate them."""
    t = (title or "").strip()
    s = (summary or "").strip()[:500]
    if t and s:
        return f"{t} - {s}"
    return t or s


async def _collapse_similar_embeddings(
    session: Session,
    ordered_ids: list[str],
    threshold: float,
) -> tuple[list[str], list[tuple[str, str, float]]]:
    """Greedy intra-topic near-dup collapse. Re-embeds title+summary on the
    fly rather than using the stored title-only vectors (see _content_text_for_dedup
    for the reason). Returns (kept_in_order, dropped_log) where each
    dropped_log entry is (dropped_id, matched_kept_id, similarity).

    Embedding failure fails open (keeps everything). Cheap: N<=20 items
    per topic, one batched call to the embeddings endpoint."""
    if len(ordered_ids) < 2:
        return list(ordered_ids), []

    rows = session.exec(
        select(NewsItem.id, NewsItem.title, NewsItem.summary).where(
            NewsItem.id.in_([_uuid(i) for i in ordered_ids])
        )
    ).all()
    meta_by_id: dict[str, tuple[Optional[str], Optional[str]]] = {
        str(r[0]): (r[1], r[2]) for r in rows
    }

    texts: list[str] = []
    order_with_text: list[str] = []
    for nid in ordered_ids:
        meta = meta_by_id.get(nid)
        if not meta:
            continue
        txt = _content_text_for_dedup(meta[0], meta[1])
        if not txt:
            continue
        order_with_text.append(nid)
        texts.append(txt)

    if not texts:
        return list(ordered_ids), []

    try:
        vectors = await emb_service.embed(texts)
    except Exception as exc:
        logger.warning(
            "select collapse: embedding %d items failed (%s); skipping",
            len(texts),
            exc,
        )
        return list(ordered_ids), []

    vec_by_id: dict[str, list[float]] = dict(zip(order_with_text, vectors))

    kept: list[str] = []
    kept_vecs: list[tuple[str, list[float]]] = []
    dropped: list[tuple[str, str, float]] = []

    for nid in ordered_ids:
        vec = vec_by_id.get(nid)
        if vec is None:
            kept.append(nid)
            continue
        matched: Optional[tuple[str, float]] = None
        for kept_id, kv in kept_vecs:
            sim = _cosine(vec, kv)
            if sim >= threshold and (matched is None or sim > matched[1]):
                matched = (kept_id, sim)
        if matched is not None:
            dropped.append((nid, matched[0], matched[1]))
        else:
            kept.append(nid)
            kept_vecs.append((nid, vec))
    return kept, dropped


class _DedupOutput(BaseModel):
    """Flat schema; same rationale as _SelectOutput."""

    drop_ids: list[str] = Field(
        default_factory=list,
        description="news_item_ids to drop as duplicates of an earlier item in the list.",
    )
    drop_reasons: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Map of dropped_id -> short one-line reason (under 80 chars). "
            "Example: 'same event as <other_id>, different outlet'."
        ),
    )


async def _llm_dedup_pass(
    topic_name: str,
    items: list[tuple[str, Optional[str], Optional[str]]],
    state: Optional[AgentState] = None,
) -> tuple[list[str], dict[str, str]]:
    """Ask the LLM to cluster same-event dupes among items that already
    passed select. Returns (drop_ids, drop_reasons). Fails open on any
    error: empty results mean 'no drops'.

    Input is (id, title, summary) tuples in the order the LLM first saw
    them. The prompt tells the LLM to keep the first of any cluster.
    """
    if len(items) < 2:
        return [], {}
    try:
        llm = get_structured_llm(_DedupOutput, temperature=0)
    except LLMUnavailable:
        return [], {}

    lines = []
    for nid, title, summary in items:
        t = (title or "").strip()
        s = (summary or "").strip()[:200]
        lines.append(f"- id={nid}  title={t!r}  summary={s!r}")
    user_msg = (
        prompts.DEDUP_INSTRUCTION.format(topic_name=topic_name) + "\n\nItems:\n" + "\n".join(lines)
    )
    try:
        result = await _invoke_with_parse_retry(
            llm,
            [
                SystemMessage(content=prompts.SYSTEM_PROMPT),
                HumanMessage(content=user_msg),
            ],
            state=state,
            stage="intra_dedup",
        )
    except LLM_CATCHABLE as exc:
        logger.warning(
            "select dedup-pass: LLM exhausted retries for topic %r (%s). "
            "Keeping all items as-is.",
            topic_name,
            exc,
        )
        return [], {}

    valid_ids = {nid for (nid, _, _) in items}
    drop_ids = [str(x) for x in (result.drop_ids or []) if str(x) in valid_ids]
    reasons = {
        str(k): str(v)[:80] for k, v in (result.drop_reasons or {}).items() if str(k) in valid_ids
    }
    return drop_ids, reasons


def _cosine(a, b) -> float:
    """Cosine similarity. Accepts list[float] or numpy.ndarray (pgvector
    returns the latter). Falsy-checks (`not a`) would raise ValueError on
    numpy arrays, hence the explicit None / length guards."""
    if a is None or b is None:
        return 0.0
    try:
        la, lb = len(a), len(b)
    except TypeError:
        return 0.0
    if la == 0 or lb == 0 or la != lb:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na**0.5 * nb**0.5)


def _checkpoint_progress(state: AgentState) -> None:
    """Flush the current meta snapshot + durable log tail. Used at sub-step
    boundaries inside LLM-heavy nodes so the live dashboard sees per-topic /
    per-story progress and logs survive an in-memory buffer eviction."""
    if not isinstance(state, dict):
        return
    issue_id = state.get("issue_id")
    if not issue_id:
        return
    meta = state.get("meta") or {}
    checkpoint_meta(issue_id, meta)
    log_buffer.flush_to_db(issue_id)
    # Throttled live report refresh at select/draft/critique sub-steps.
    report_service.write_if_due(state, final=False)


async def select_stories(state: AgentState) -> AgentState:
    from app.config import settings

    if not chat_llm_configured():
        _warn(
            state,
            "chat LLM not configured; select is a passthrough (all kept items forwarded, first item per topic is lead).",
        )
        return _select_passthrough(state)

    batch_size = max(1, settings.select_batch_size)
    # temperature=0 on select: filtering is a decision task, not creative
    # writing. Same prompt should give the same picks run-to-run.
    llm = get_structured_llm(_SelectOutput, temperature=0)
    selected_by_topic: dict[str, list[str]] = {}
    lead_by_topic: dict[str, str] = {}

    with Session(engine) as s:
        topics = s.exec(
            select(Topic).where(Topic.id.in_([_uuid(t) for t in state["topic_ids"]]))
        ).all()
        kept_ids = set(state.get("kept_ids", []))
        config = s.get(NewspaperConfig, _uuid(state["config_id"]))

        # Initialize live progress for the dashboard. per_topic starts with
        # one "pending" entry per topic; each gets flipped to running/done
        # as we iterate. Sub-step updates (current_batch, current_phase)
        # land on meta['select']['progress'] and checkpoint after each write.
        meta_select = state.setdefault("meta", {}).setdefault("select", {})
        progress = meta_select["progress"] = {
            "topics_total": len(topics),
            "topics_done": 0,
            "current_topic_id": None,
            "current_topic_name": None,
            "current_topic_index": 0,
            "current_phase": None,
            "current_batch": 0,
            "current_batch_total": 0,
            "phase_started_at": None,
            "per_topic": [
                {"topic_id": str(t.id), "topic_name": t.name, "status": "pending"} for t in topics
            ],
        }
        _checkpoint_progress(state)

        for topic_index, topic in enumerate(topics, start=1):
            candidates = s.exec(
                select(NewsItem)
                .where(NewsItem.topic_id == topic.id)
                .where(NewsItem.id.in_([_uuid(i) for i in kept_ids]))
            ).all()

            # Mark this topic's per_topic row as running for the UI.
            for row in progress["per_topic"]:
                if row.get("topic_id") == str(topic.id):
                    row["status"] = "running"
                    break
            progress["current_topic_id"] = str(topic.id)
            progress["current_topic_name"] = topic.name
            progress["current_topic_index"] = topic_index
            progress["current_phase"] = None
            progress["current_batch"] = 0
            progress["current_batch_total"] = 0
            progress["phase_started_at"] = now_utc().isoformat()
            _checkpoint_progress(state)

            if not candidates:
                for row in progress["per_topic"]:
                    if row.get("topic_id") == str(topic.id):
                        row["status"] = "done"
                        row["candidates"] = 0
                        row["batches"] = 0
                        row["kept"] = 0
                        row["elapsed_ms"] = 0
                        break
                progress["topics_done"] += 1
                _checkpoint_progress(state)
                continue

            topic_t0 = time.perf_counter()
            keep_ordered: list[str] = []
            seen_keep: set[str] = set()
            lead: Optional[str] = None
            n_batches = (len(candidates) + batch_size - 1) // batch_size

            for batch_idx in range(n_batches):
                chunk = candidates[batch_idx * batch_size : (batch_idx + 1) * batch_size]

                progress["current_phase"] = "batch"
                progress["current_batch"] = batch_idx + 1
                progress["current_batch_total"] = n_batches
                progress["phase_started_at"] = now_utc().isoformat()
                _checkpoint_progress(state)
                logger.info(
                    "agent select: topic=%r (%d/%d) batch %d/%d start (candidates=%d)",
                    topic.name,
                    topic_index,
                    len(topics),
                    batch_idx + 1,
                    n_batches,
                    len(chunk),
                )
                batch_t0 = time.perf_counter()

                try:
                    result = await _select_chunk(llm, topic.name, chunk, state=state)
                except LLM_CATCHABLE as exc:
                    # DROP the batch. _invoke_with_parse_retry already gave
                    # the call 3 chances (mix of timeouts + parse failures).
                    # Passthrough would leak unfiltered junk into the draft
                    # phase; dropping preserves quality at the cost of some
                    # potentially-good items. The user explicitly prefers drop.
                    _warn(
                        state,
                        f"select LLM exhausted retries for topic {topic.name} "
                        f"(batch {batch_idx + 1}/{n_batches}, {len(chunk)} items, "
                        f"{type(exc).__name__}): {exc}. Dropping this batch.",
                    )
                    for item in chunk:
                        report_service.record_event(
                            state,
                            news_item_id=str(item.id),
                            step="select",
                            status="dropped",
                            topic_id=str(topic.id),
                            topic_name=topic.name,
                            reason=(
                                "Quality-filter LLM call failed after retries; "
                                "the whole batch was dropped."
                            ),
                            reason_source="system",
                            data={
                                "batch_index": batch_idx + 1,
                                "batch_total": n_batches,
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                        )
                    continue

                # Merge flat schema: union keep_ids preserving order; take
                # the first batch's lead_id that falls inside its own keep_ids.
                # The prompt asks the LLM for at most select_max_kept_per_batch
                # best-ranked items; this slice is a safety net that respects
                # the LLM's own ordering (drops tail, not head).
                chunk_keeps = set(str(c.id) for c in chunk)
                chunk_by_id = {str(c.id): c for c in chunk}
                per_batch_cap = max(1, settings.select_max_kept_per_batch)
                kept_from_chunk = 0
                accepted_this_batch: set[str] = set()
                cap_trimmed_this_batch: set[str] = set()
                keep_reasons = {
                    str(k): str(v)[:120]
                    for k, v in (result.keep_reasons or {}).items()
                    if str(k) in chunk_keeps
                }
                for nid in result.keep_ids or []:
                    nid_str = str(nid)
                    if nid_str not in chunk_keeps:
                        continue
                    if kept_from_chunk >= per_batch_cap:
                        # Accepted by LLM but trimmed by the cap; record.
                        n = chunk_by_id.get(nid_str)
                        _record_drop(
                            state,
                            news_item_id=nid_str,
                            stage="select_cap",
                            topic_id=str(topic.id),
                            reason=f"exceeded per-batch cap of {per_batch_cap}",
                            snapshot=_snapshot_news_item(n) if n else None,
                        )
                        cap_trimmed_this_batch.add(nid_str)
                        report_service.record_event(
                            state,
                            news_item_id=nid_str,
                            step="select_cap",
                            status="dropped",
                            topic_id=str(topic.id),
                            topic_name=topic.name,
                            reason=f"Exceeded the per-batch cap of {per_batch_cap}.",
                            reason_source="system",
                            data={
                                "batch_index": batch_idx + 1,
                                "batch_total": n_batches,
                                "max_kept_per_batch": per_batch_cap,
                            },
                        )
                        continue
                    if nid_str not in seen_keep:
                        seen_keep.add(nid_str)
                        keep_ordered.append(nid_str)
                        kept_from_chunk += 1
                        report_service.record_event(
                            state,
                            news_item_id=nid_str,
                            step="select",
                            status="kept",
                            topic_id=str(topic.id),
                            topic_name=topic.name,
                            reason=keep_reasons.get(
                                nid_str, "Selected by the quality filter."
                            ),
                            reason_source="ai" if nid_str in keep_reasons else "system",
                            data={
                                "batch_index": batch_idx + 1,
                                "batch_total": n_batches,
                                "rank": kept_from_chunk,
                            },
                        )
                    accepted_this_batch.add(nid_str)
                if lead is None and result.lead_id:
                    lead_str = str(result.lead_id)
                    if lead_str in seen_keep:
                        lead = lead_str

                # Record everyone in this chunk the LLM chose to drop.
                llm_reasons = {
                    str(k): str(v)[:80]
                    for k, v in (result.drop_reasons or {}).items()
                    if str(k) in chunk_keeps
                }
                for c in chunk:
                    nid_str = str(c.id)
                    if nid_str in accepted_this_batch:
                        continue
                    reason = llm_reasons.get(
                        nid_str, "dropped by quality filter; no reason given"
                    )
                    _record_drop(
                        state,
                        news_item_id=nid_str,
                        stage="select_llm",
                        topic_id=str(topic.id),
                        reason=reason,
                        snapshot=_snapshot_news_item(c),
                    )
                    # Cap-trimmed items already have a terminal select_cap
                    # event; do not add a duplicate report entry.
                    if nid_str in cap_trimmed_this_batch:
                        continue
                    report_service.record_event(
                        state,
                        news_item_id=nid_str,
                        step="select",
                        status="dropped",
                        topic_id=str(topic.id),
                        topic_name=topic.name,
                        reason=reason,
                        reason_source="ai" if nid_str in llm_reasons else "system",
                        data={
                            "batch_index": batch_idx + 1,
                            "batch_total": n_batches,
                        },
                    )

                batch_elapsed = time.perf_counter() - batch_t0
                logger.info(
                    "agent select: topic=%r (%d/%d) batch %d/%d kept=%d elapsed=%.1fs",
                    topic.name,
                    topic_index,
                    len(topics),
                    batch_idx + 1,
                    n_batches,
                    len(accepted_this_batch),
                    batch_elapsed,
                )
                _checkpoint_progress(state)

            # Fallback: if no batch flagged a lead, promote the first kept item.
            if lead is None and keep_ordered:
                lead = keep_ordered[0]

            # Intra-topic near-dup collapse: two passes, LLM FIRST.
            # Pass 1: LLM dedup sees all items with title+summary and
            #   clusters by semantic event. Picks the most substantive item
            #   per cluster as the representative (specific story beats
            #   aggregator landing page). Cross-batch view the batched
            #   select can't have.
            # Pass 2: embedding cosine at settings.select_intra_sim_threshold
            #   as a cheap safety net on the LLM's survivors, using a
            #   re-embed of title+summary (the stored title-only vectors
            #   cluster too loosely on short Spanish outlet paraphrases).
            before_count = len(keep_ordered)

            dedup_meta_rows = (
                s.exec(
                    select(NewsItem).where(NewsItem.id.in_([_uuid(i) for i in keep_ordered]))
                ).all()
                if keep_ordered
                else []
            )
            dedup_by_id = {str(n.id): n for n in dedup_meta_rows}

            if len(keep_ordered) >= 2:
                progress["current_phase"] = "intra_dedup_llm"
                progress["current_batch"] = 0
                progress["current_batch_total"] = 0
                progress["phase_started_at"] = now_utc().isoformat()
                _checkpoint_progress(state)
                logger.info(
                    "agent select: topic=%r (%d/%d) intra-dedup (llm) start",
                    topic.name,
                    topic_index,
                    len(topics),
                )
                dedup_t0 = time.perf_counter()

                items_for_llm = [
                    (
                        nid,
                        dedup_by_id[nid].title if nid in dedup_by_id else None,
                        dedup_by_id[nid].summary if nid in dedup_by_id else None,
                    )
                    for nid in keep_ordered
                ]
                drop_ids, drop_reasons = await _llm_dedup_pass(
                    topic.name, items_for_llm, state=state
                )
                if drop_ids:
                    drop_set = set(drop_ids)
                    for nid in drop_ids:
                        n = dedup_by_id.get(nid)
                        logger.info(
                            "select collapse (llm): topic=%r dropped=%s",
                            topic.name,
                            nid,
                        )
                        reason = drop_reasons.get(nid, "duplicate of another kept item")
                        _record_drop(
                            state,
                            news_item_id=nid,
                            stage="intra_dedup_llm",
                            topic_id=str(topic.id),
                            reason=reason,
                            snapshot=_snapshot_news_item(n) if n else None,
                        )
                        report_service.record_event(
                            state,
                            news_item_id=nid,
                            step="intra_dedup_llm",
                            status="dropped",
                            topic_id=str(topic.id),
                            topic_name=topic.name,
                            reason=reason,
                            reason_source="ai",
                        )
                    keep_ordered = [nid for nid in keep_ordered if nid not in drop_set]
                for nid in keep_ordered:
                    report_service.record_event(
                        state,
                        news_item_id=nid,
                        step="intra_dedup_llm",
                        status="kept",
                        topic_id=str(topic.id),
                        topic_name=topic.name,
                        reason="Not flagged as a duplicate by the AI dedup pass.",
                        reason_source="system",
                    )

                logger.info(
                    "agent select: topic=%r (%d/%d) intra-dedup (llm) dropped=%d elapsed=%.1fs",
                    topic.name,
                    topic_index,
                    len(topics),
                    len(drop_ids or []),
                    time.perf_counter() - dedup_t0,
                )
                _checkpoint_progress(state)

            progress["current_phase"] = "intra_dedup_emb"
            progress["phase_started_at"] = now_utc().isoformat()
            _checkpoint_progress(state)

            keep_ordered, collapsed = await _collapse_similar_embeddings(
                s,
                keep_ordered,
                settings.select_intra_sim_threshold,
            )
            dropped_emb_ids = {pair[0] for pair in (collapsed or [])}
            if collapsed:
                for dropped_id, kept_id, sim in collapsed:
                    n = dedup_by_id.get(dropped_id)
                    logger.info(
                        "select collapse (emb): topic=%r dropped=%s kept=%s sim=%.3f",
                        topic.name,
                        dropped_id,
                        kept_id,
                        sim,
                    )
                    reason = f"near-duplicate of {kept_id} (sim={sim:.3f})"
                    _record_drop(
                        state,
                        news_item_id=dropped_id,
                        stage="intra_dedup_emb",
                        topic_id=str(topic.id),
                        reason=reason,
                        snapshot=_snapshot_news_item(n) if n else None,
                    )
                    report_service.record_event(
                        state,
                        news_item_id=dropped_id,
                        step="intra_dedup_emb",
                        status="dropped",
                        topic_id=str(topic.id),
                        topic_name=topic.name,
                        reason=reason,
                        reason_source="system",
                        data={"similarity": sim, "matched_kept_id": kept_id},
                    )
            for nid in keep_ordered:
                if nid in dropped_emb_ids:
                    continue
                report_service.record_event(
                    state,
                    news_item_id=nid,
                    step="intra_dedup_emb",
                    status="kept",
                    topic_id=str(topic.id),
                    topic_name=topic.name,
                    reason="Below the embedding near-duplicate threshold.",
                    reason_source="system",
                )
            logger.info(
                "agent select: topic=%r (%d/%d) intra-dedup (emb) dropped=%d",
                topic.name,
                topic_index,
                len(topics),
                len(collapsed or []),
            )

            if lead and lead not in keep_ordered:
                lead = keep_ordered[0] if keep_ordered else None

            # Optional per-config hard cap on how many stories actually
            # make it to draft for this topic.
            if config and config.max_stories_per_topic:
                trimmed = keep_ordered[config.max_stories_per_topic :]
                keep_ordered = keep_ordered[: config.max_stories_per_topic]
                for nid in trimmed:
                    n = dedup_by_id.get(nid)
                    reason = f"exceeded max_stories_per_topic={config.max_stories_per_topic}"
                    _record_drop(
                        state,
                        news_item_id=nid,
                        stage="select_topic_cap",
                        topic_id=str(topic.id),
                        reason=reason,
                        snapshot=_snapshot_news_item(n) if n else None,
                    )
                    report_service.record_event(
                        state,
                        news_item_id=nid,
                        step="topic_cap",
                        status="dropped",
                        topic_id=str(topic.id),
                        topic_name=topic.name,
                        reason=reason,
                        reason_source="system",
                        data={"max_stories_per_topic": config.max_stories_per_topic},
                    )
                if lead and lead not in keep_ordered:
                    lead = keep_ordered[0] if keep_ordered else None

            selected_by_topic[str(topic.id)] = keep_ordered
            if lead:
                lead_by_topic[str(topic.id)] = lead

            state.setdefault("meta", {}).setdefault("select", {}).setdefault("topics", {})[
                str(topic.id)
            ] = {
                "topic_name": topic.name,
                "candidates": len(candidates),
                "batches": n_batches,
                "kept": len(keep_ordered),
                "collapsed_total": before_count - len(keep_ordered),
                "has_lead": bool(lead),
            }

            topic_elapsed_ms = int((time.perf_counter() - topic_t0) * 1000)
            logger.info(
                "agent select: topic=%r candidates=%d batches=%d kept=%d collapsed=%d lead=%s",
                topic.name,
                len(candidates),
                n_batches,
                len(keep_ordered),
                before_count - len(keep_ordered),
                "yes" if lead else "no",
            )

            for row in progress["per_topic"]:
                if row.get("topic_id") == str(topic.id):
                    row["status"] = "done"
                    row["candidates"] = len(candidates)
                    row["batches"] = n_batches
                    row["kept"] = len(keep_ordered)
                    row["elapsed_ms"] = topic_elapsed_ms
                    break
            progress["topics_done"] += 1
            progress["current_phase"] = None
            progress["current_batch"] = 0
            progress["current_batch_total"] = 0
            progress["phase_started_at"] = None
            _checkpoint_progress(state)

        # All topics done. Clear the "currently processing" fields so the
        # UI renders the final per_topic list without an orphan active row.
        progress["current_topic_id"] = None
        progress["current_topic_name"] = None
        progress["current_topic_index"] = 0

    state["selected_by_topic"] = selected_by_topic
    state["lead_by_topic"] = lead_by_topic
    total_kept = sum(len(v) for v in selected_by_topic.values())
    meta_select = state.setdefault("meta", {}).setdefault("select", {})
    meta_select["total_kept"] = total_kept
    meta_select["topics_with_keeps"] = len(selected_by_topic)
    logger.info("agent select: %d topics, %d items total", len(selected_by_topic), total_kept)
    return state


def _select_passthrough(state: AgentState) -> AgentState:
    """When the LLM isn't available, keep everything dedupe produced and
    pick the first per topic as the lead."""
    selected_by_topic: dict[str, list[str]] = {}
    lead_by_topic: dict[str, str] = {}
    with Session(engine) as s:
        kept_ids = state.get("kept_ids", [])
        if kept_ids:
            kept_items = s.exec(
                select(NewsItem).where(NewsItem.id.in_([_uuid(i) for i in kept_ids]))
            ).all()
            for item in kept_items:
                if item.topic_id is None:
                    continue
                tid = str(item.topic_id)
                selected_by_topic.setdefault(tid, []).append(str(item.id))
            for tid, ids in selected_by_topic.items():
                lead_by_topic[tid] = ids[0]
                for nid in ids:
                    report_service.record_event(
                        state,
                        news_item_id=nid,
                        step="select",
                        status="kept",
                        topic_id=tid,
                        reason="Chat LLM not configured; passthrough kept all candidates.",
                        reason_source="system",
                    )
    state["selected_by_topic"] = selected_by_topic
    state["lead_by_topic"] = lead_by_topic
    return state


# ---------------------------------------------------------------------------
# Node 6: fetch_articles: retrieve full web text for deduped candidates before selection


async def fetch_articles(state: AgentState) -> AgentState:
    from app.services.article_extractor import fetch_article_text

    kept_ids = state.get("kept_ids") or []
    all_item_ids: list[str] = []
    for i in kept_ids:
        if i not in all_item_ids:
            all_item_ids.append(i)

    stories_total = len(all_item_ids)
    meta_fetch = state.setdefault("meta", {}).setdefault("fetch_articles", {})
    progress = meta_fetch["progress"] = {
        "stories_total": stories_total,
        "stories_done": 0,
        "current_title": None,
        "current_url": None,
    }
    _checkpoint_progress(state)

    if not all_item_ids:
        meta_fetch.update(
            {"attempted": 0, "cached": 0, "fetched_ok": 0, "fallback_summary": 0}
        )
        return state

    cached = 0
    fetched_ok = 0
    fallback_summary = 0
    fetch_results: dict[str, dict] = {}

    with Session(engine) as s:
        items = s.exec(
            select(NewsItem).where(NewsItem.id.in_([_uuid(i) for i in all_item_ids]))
        ).all()
        items_by_id = {str(item.id): item for item in items}

    sem = asyncio.Semaphore(settings.article_fetch_concurrency)

    async def _process_item(item_id: str):
        nonlocal cached, fetched_ok, fallback_summary
        item = items_by_id.get(item_id)
        if not item:
            return

        progress["current_title"] = (item.title or "")[:80]
        progress["current_url"] = item.url
        _checkpoint_progress(state)

        if item.article_text and len(item.article_text.strip()) >= 50:
            cached += 1
            fetch_results[item_id] = {
                "status": "cached",
                "text": item.article_text,
                "char_count": len(item.article_text),
                "error": None,
            }
            progress["stories_done"] += 1
            _checkpoint_progress(state)
            return

        text: Optional[str] = None
        async with sem:
            try:
                text = await fetch_article_text(
                    item.url,
                    timeout=settings.article_fetch_timeout_seconds,
                )
                logger.info("DEBUG nodes.py fetch_article_text from %s returns %s", item.url, text)
            except Exception as exc:
                logger.warning("fetch_articles error for %s: %s", item.url, exc)
                text = None

        with Session(engine) as s:
            db_item = s.get(NewsItem, item.id)
            if db_item:
                if text:
                    db_item.article_text = text
                    db_item.article_fetched_at = now_utc()
                    db_item.article_fetch_status = "ok"
                    fetched_ok += 1
                    fetch_results[item_id] = {
                        "status": "fetched",
                        "text": text,
                        "char_count": len(text),
                        "error": None,
                    }
                else:
                    db_item.article_fetch_status = "fallback_summary"
                    db_item.article_fetched_at = now_utc()
                    fallback_summary += 1
                    fetch_results[item_id] = {
                        "status": "fallback",
                        "text": None,
                        "char_count": 0,
                        "error": "Fetch or extraction failed; used the stored summary.",
                    }
                s.add(db_item)
                s.commit()

        progress["stories_done"] += 1
        _checkpoint_progress(state)

    tasks = [_process_item(iid) for iid in all_item_ids]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    # Record in original candidate order rather than task completion order.
    for item_id in all_item_ids:
        item = items_by_id.get(item_id)
        if item is not None:
            report_service.ensure_article(state, item, synthesized=True)
        result = fetch_results.get(item_id)
        if not result:
            result = {
                "status": "failed",
                "text": None,
                "char_count": 0,
                "error": "Fetch did not produce a result.",
            }
        report_service.record_event(
            state,
            news_item_id=item_id,
            step="article_fetch",
            status=result["status"],
            topic_id=str(item.topic_id) if item and item.topic_id else None,
            reason=result.get("error"),
            reason_source="system",
            data={
                "text": result.get("text"),
                "char_count": result.get("char_count", 0),
                "truncated": bool(
                    settings.article_max_characters > 0
                    and result.get("char_count", 0) >= settings.article_max_characters
                ),
                "error": result.get("error"),
            },
        )

    progress["current_title"] = None
    progress["current_url"] = None
    _checkpoint_progress(state)

    meta_fetch.update(
        {
            "attempted": stories_total,
            "cached": cached,
            "fetched_ok": fetched_ok,
            "fallback_summary": fallback_summary,
        }
    )
    logger.info(
        "agent fetch_articles: %d total (%d cached, %d fetched, %d fallback to summary)",
        stories_total,
        cached,
        fetched_ok,
        fallback_summary,
    )
    return state


# ---------------------------------------------------------------------------
# Node 7: draft


class _DraftOutput(BaseModel):
    rendered_title: str = Field(description="Replacement headline, under 70 chars.")
    body: str = Field(description="100 to 150 word news summary in cause->effect form.")


async def draft(state: AgentState) -> AgentState:
    if not chat_llm_configured():
        _warn(state, "chat LLM not configured; draft is a passthrough (uses source title/summary).")
        return _draft_passthrough(state)

    llm = get_structured_llm(_DraftOutput)
    drafts: list[DraftDict] = []
    attempted = 0
    failures = 0

    # Count how many stories draft will process across all topics so the UI
    # can render "Story 3/23" without walking state itself.
    stories_total = sum(len(ids) for ids in (state.get("selected_by_topic") or {}).values())
    meta_draft = state.setdefault("meta", {}).setdefault("draft", {})
    progress = meta_draft["progress"] = {
        "stories_total": stories_total,
        "stories_done": 0,
        "current_story_index": 0,
        "current_topic_name": None,
        "current_title": None,
        "story_started_at": None,
    }
    _checkpoint_progress(state)

    with Session(engine) as s:
        config = s.get(NewspaperConfig, _uuid(state["config_id"]))
        lang_code = getattr(config, "language", "en") if config else "en"
        lang_name = language_name(lang_code)
        topics_by_id = {
            str(t.id): t
            for t in s.exec(
                select(Topic).where(Topic.id.in_([_uuid(i) for i in state["topic_ids"]]))
            ).all()
        }
        for topic_id, item_ids in state.get("selected_by_topic", {}).items():
            topic = topics_by_id.get(topic_id)
            if topic is None or not item_ids:
                continue
            items = s.exec(
                select(NewsItem).where(NewsItem.id.in_([_uuid(i) for i in item_ids]))
            ).all()
            lead_id = state.get("lead_by_topic", {}).get(topic_id)

            for item in items:
                attempted += 1
                progress["current_story_index"] = attempted
                progress["current_topic_name"] = topic.name
                progress["current_title"] = (item.title or "")[:80]
                progress["story_started_at"] = now_utc().isoformat()
                _checkpoint_progress(state)
                logger.info(
                    "agent draft: story %d/%d (topic=%r) start",
                    attempted,
                    stories_total,
                    topic.name,
                )
                story_t0 = time.perf_counter()

                source_kind = "article_text" if item.article_text else "summary"
                source_text = (
                    (item.article_text or item.summary or "")[: settings.article_max_characters]
                )
                report_service.ensure_article(state, item, topic_name=topic.name, synthesized=True)
                report_service.record_event(
                    state,
                    news_item_id=str(item.id),
                    step="draft_input",
                    status="info",
                    topic_id=topic_id,
                    topic_name=topic.name,
                    reason=f"Draft source: {source_kind}.",
                    reason_source="system",
                    data={
                        "source_kind": source_kind,
                        "source_text": source_text,
                        "char_count": len(source_text),
                        "title": item.title or "",
                        "url": report_service.safe_url(item.url),
                    },
                )
                user_msg = prompts.DRAFT_INSTRUCTION.format(
                    topic_name=topic.name,
                    title=item.title or "",
                    url=item.url,
                    summary=source_text,
                    language_name=lang_name,
                )
                try:
                    out: _DraftOutput = await _invoke_with_parse_retry(
                        llm,
                        [
                            SystemMessage(content=prompts.SYSTEM_PROMPT),
                            HumanMessage(content=user_msg),
                        ],
                        state=state,
                        stage="draft",
                    )
                except LLM_CATCHABLE as exc:
                    failures += 1
                    _warn(
                        state,
                        f"draft LLM failed for {item.id} ({type(exc).__name__}): {exc}. Skipping item.",
                    )
                    report_service.record_event(
                        state,
                        news_item_id=str(item.id),
                        step="draft",
                        status="failed",
                        topic_id=topic_id,
                        topic_name=topic.name,
                        reason=f"{type(exc).__name__}: {exc}",
                        reason_source="system",
                    )
                    logger.info(
                        "agent draft: story %d/%d failed in %.1fs",
                        attempted,
                        stories_total,
                        time.perf_counter() - story_t0,
                    )
                    progress["stories_done"] += 1
                    _checkpoint_progress(state)
                    continue

                rendered_title = out.rendered_title.strip()
                rendered_body = out.body.strip()
                is_lead = str(item.id) == lead_id
                drafts.append(
                    DraftDict(
                        news_item_id=str(item.id),
                        topic_id=topic_id,
                        topic_name=topic.name,
                        rendered_title=rendered_title,
                        body=rendered_body,
                        is_lead=is_lead,
                        original_title=item.title or "",
                        original_url=item.url,
                    )
                )
                report_service.record_event(
                    state,
                    news_item_id=str(item.id),
                    step="draft",
                    status="ok",
                    topic_id=topic_id,
                    topic_name=topic.name,
                    reason="Draft produced.",
                    reason_source="system",
                    data={
                        "rendered_title": rendered_title,
                        "body": rendered_body,
                        "is_lead": is_lead,
                    },
                )
                logger.info(
                    "agent draft: story %d/%d succeeded in %.1fs",
                    attempted,
                    stories_total,
                    time.perf_counter() - story_t0,
                )
                progress["stories_done"] += 1
                _checkpoint_progress(state)

    progress["current_story_index"] = 0
    progress["current_topic_name"] = None
    progress["current_title"] = None
    progress["story_started_at"] = None
    _checkpoint_progress(state)

    state["drafts"] = drafts
    meta_draft.update(
        {
            "attempted": attempted,
            "succeeded": len(drafts),
            "failures": failures,
        }
    )
    logger.info("agent draft: %d drafts produced", len(drafts))
    logger.info("DEBUG nodes.py state %s", drafts)
    return state


def _draft_passthrough(state: AgentState) -> AgentState:
    drafts: list[DraftDict] = []
    with Session(engine) as s:
        topics_by_id = {
            str(t.id): t
            for t in s.exec(
                select(Topic).where(Topic.id.in_([_uuid(i) for i in state["topic_ids"]]))
            ).all()
        }
        for topic_id, item_ids in state.get("selected_by_topic", {}).items():
            topic = topics_by_id.get(topic_id)
            if topic is None or not item_ids:
                continue
            items = s.exec(
                select(NewsItem).where(NewsItem.id.in_([_uuid(i) for i in item_ids]))
            ).all()
            lead_id = state.get("lead_by_topic", {}).get(topic_id)
            for item in items:
                source_text = (item.article_text or item.summary or "")[:1500]
                rendered_title = item.title or ""
                rendered_body = source_text
                is_lead = str(item.id) == lead_id
                drafts.append(
                    DraftDict(
                        news_item_id=str(item.id),
                        topic_id=topic_id,
                        topic_name=topic.name,
                        rendered_title=rendered_title,
                        body=rendered_body,
                        is_lead=is_lead,
                        original_title=item.title or "",
                        original_url=item.url,
                    )
                )
                report_service.record_event(
                    state,
                    news_item_id=str(item.id),
                    step="draft_input",
                    status="info",
                    topic_id=topic_id,
                    topic_name=topic.name,
                    reason="Chat LLM not configured; passthrough draft.",
                    reason_source="system",
                    data={
                        "source_kind": "article_text" if item.article_text else "summary",
                        "source_text": source_text,
                        "char_count": len(source_text),
                    },
                )
                report_service.record_event(
                    state,
                    news_item_id=str(item.id),
                    step="draft",
                    status="ok",
                    topic_id=topic_id,
                    topic_name=topic.name,
                    reason="Passthrough used the source text as the draft.",
                    reason_source="system",
                    data={
                        "rendered_title": rendered_title,
                        "body": rendered_body,
                        "is_lead": is_lead,
                    },
                )
    state["drafts"] = drafts
    return state


# ---------------------------------------------------------------------------
# Node 7: editorial + critique (lightweight)


class _CritiqueOutput(BaseModel):
    needs_rewrite: bool
    body: str


def _has_voice_violation(body: str, language: str = "en") -> bool:
    """Mechanical pre-check for the critique node. Two checks are
    language-agnostic (exclamation marks, word count); the hype-word
    regex is per-language, pulled from app.agent.languages."""
    if not body:
        return False
    if "!" in body:
        return True
    lowered = body.lower()
    words = hype_words(language)
    if words and any(w in lowered for w in words):
        return True
    wc = len(body.split())
    if wc < 80 or wc > 180:
        return True
    return False


async def critique(state: AgentState) -> AgentState:
    drafts = state.get("drafts", [])
    if not drafts:
        state.setdefault("meta", {})["critique"] = {
            "checked": 0,
            "flagged": 0,
            "rewrites": 0,
            "failures": 0,
            "llm": False,
        }
        return state

    llm_available = chat_llm_configured()
    rewrites = 0
    flagged = 0
    failures = 0

    meta_critique = state.setdefault("meta", {}).setdefault("critique", {})
    progress = meta_critique["progress"] = {
        "drafts_total": len(drafts),
        "drafts_checked": 0,
        "rewrites_done": 0,
        "current_draft_index": 0,
        "current_phase": None,
    }
    _checkpoint_progress(state)

    with Session(engine) as s:
        config = s.get(NewspaperConfig, _uuid(state["config_id"]))
    lang_code = getattr(config, "language", "en") if config else "en"
    lang_name = language_name(lang_code)

    if llm_available:
        # temperature=0 on critique: the rewrite should sharpen predictably,
        # not introduce fresh variation on top of an already-flagged body.
        llm = get_structured_llm(_CritiqueOutput, temperature=0)
        for idx, d in enumerate(drafts, start=1):
            progress["current_draft_index"] = idx
            progress["current_phase"] = "check"
            _checkpoint_progress(state)

            if not _has_voice_violation(d.get("body", ""), lang_code):
                report_service.record_event(
                    state,
                    news_item_id=d.get("news_item_id"),
                    step="critique",
                    status="ok",
                    topic_id=d.get("topic_id"),
                    topic_name=d.get("topic_name"),
                    reason="No voice violation detected.",
                    reason_source="system",
                    data={"body": d.get("body", "")},
                )
                progress["drafts_checked"] += 1
                logger.info("agent critique: draft %d/%d clean (no rewrite)", idx, len(drafts))
                _checkpoint_progress(state)
                continue
            flagged += 1
            progress["current_phase"] = "rewrite"
            _checkpoint_progress(state)
            logger.info("agent critique: draft %d/%d rewrite required", idx, len(drafts))
            rewrite_t0 = time.perf_counter()
            before_body = d.get("body", "")
            user_msg = prompts.CRITIQUE_INSTRUCTION.format(
                body=before_body,
                language_name=lang_name,
            )
            try:
                out: _CritiqueOutput = await _invoke_with_parse_retry(
                    llm,
                    [SystemMessage(content=prompts.SYSTEM_PROMPT), HumanMessage(content=user_msg)],
                    state=state,
                    stage="critique",
                )
                if out.needs_rewrite:
                    d["body"] = out.body.strip()
                    rewrites += 1
                    progress["rewrites_done"] = rewrites
                report_service.record_event(
                    state,
                    news_item_id=d.get("news_item_id"),
                    step="critique",
                    status="rewritten" if out.needs_rewrite else "ok",
                    topic_id=d.get("topic_id"),
                    topic_name=d.get("topic_name"),
                    reason=(
                        "Voice violation detected; body rewritten."
                        if out.needs_rewrite
                        else "Flagged mechanically, but the review found no rewrite necessary."
                    ),
                    reason_source="ai",
                    data={
                        "before": before_body,
                        "after": d.get("body", ""),
                        "body": d.get("body", ""),
                    },
                )
                logger.info(
                    "agent critique: draft %d/%d rewrite completed in %.1fs",
                    idx,
                    len(drafts),
                    time.perf_counter() - rewrite_t0,
                )
            except LLM_CATCHABLE as exc:
                failures += 1
                _warn(state, f"critique LLM failed ({type(exc).__name__}): {exc}")
                report_service.record_event(
                    state,
                    news_item_id=d.get("news_item_id"),
                    step="critique",
                    status="failed",
                    topic_id=d.get("topic_id"),
                    topic_name=d.get("topic_name"),
                    reason=f"Critique LLM failed: {type(exc).__name__}: {exc}",
                    reason_source="system",
                    data={"before": before_body, "after": before_body, "body": before_body},
                )
            progress["drafts_checked"] += 1
            _checkpoint_progress(state)
    else:
        for d in drafts:
            report_service.record_event(
                state,
                news_item_id=d.get("news_item_id"),
                step="critique",
                status="skipped",
                topic_id=d.get("topic_id"),
                topic_name=d.get("topic_name"),
                reason="Chat LLM not configured; critique skipped.",
                reason_source="system",
            )

    progress["current_draft_index"] = 0
    progress["current_phase"] = None
    _checkpoint_progress(state)

    meta_critique.update(
        {
            "checked": len(drafts),
            "flagged": flagged,
            "rewrites": rewrites,
            "failures": failures,
            "llm": llm_available,
        }
    )
    logger.info("agent critique: %d rewrites (llm=%s)", rewrites, llm_available)
    return state


# ---------------------------------------------------------------------------
# Node 8: editorial


class _EditorialOutput(BaseModel):
    editorial: str = Field(description="Up to 200 word editorial.")


async def editorial(state: AgentState) -> AgentState:
    meta_ed = state.setdefault("meta", {}).setdefault("editorial", {})
    if not chat_llm_configured() or not state.get("drafts"):
        meta_ed["status"] = "skipped"
        meta_ed["progress"] = {"status": "skipped", "started_at": None, "topics_count": 0}
        report_service.record_global(
            state,
            "editorial",
            "skipped",
            reason="No drafts or chat LLM not configured.",
        )
        return state

    leads: list[str] = []
    topic_names: list[str] = []
    lang_code = "en"
    with Session(engine) as s:
        config = s.get(NewspaperConfig, _uuid(state["config_id"]))
        lang_code = getattr(config, "language", "en") if config else "en"
        topics_by_id = {
            str(t.id): t
            for t in s.exec(
                select(Topic).where(Topic.id.in_([_uuid(i) for i in state["topic_ids"]]))
            ).all()
        }
        for d in state["drafts"]:
            if d.get("is_lead"):
                leads.append(d.get("rendered_title", ""))
                tname = topics_by_id.get(d.get("topic_id", ""))
                if tname:
                    topic_names.append(tname.name)

    if not leads:
        meta_ed["status"] = "skipped"
        meta_ed["reason"] = "no leads"
        meta_ed["progress"] = {"status": "skipped", "started_at": None, "topics_count": 0}
        report_service.record_global(state, "editorial", "skipped", reason="No lead stories.")
        return state

    progress = meta_ed["progress"] = {
        "status": "running",
        "started_at": now_utc().isoformat(),
        "topics_count": len(topic_names),
    }
    _checkpoint_progress(state)
    logger.info("agent editorial: start (%d topics)", len(topic_names))
    editorial_t0 = time.perf_counter()

    llm = get_structured_llm(_EditorialOutput)
    user_msg = prompts.EDITORIAL_INSTRUCTION.format(
        topic_names=", ".join(topic_names) or "(none)",
        lead_headlines="; ".join(leads),
        language_name=language_name(lang_code),
    )
    try:
        out: _EditorialOutput = await _invoke_with_parse_retry(
            llm,
            [SystemMessage(content=prompts.SYSTEM_PROMPT), HumanMessage(content=user_msg)],
            state=state,
            stage="editorial",
        )
        state["editorial"] = out.editorial.strip()
        meta_ed["status"] = "ok"
        meta_ed["word_count"] = len(state["editorial"].split())
        progress["status"] = "done"
        report_service.record_global(
            state,
            "editorial",
            "ok",
            reason="Editorial drafted.",
            data={"text": state["editorial"], "word_count": meta_ed["word_count"]},
        )
        logger.info(
            "agent editorial: ok in %.1fs (%d words)",
            time.perf_counter() - editorial_t0,
            meta_ed["word_count"],
        )
    except LLM_CATCHABLE as exc:
        meta_ed["status"] = "failed"
        meta_ed["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        progress["status"] = "failed"
        _warn(state, f"editorial LLM failed ({type(exc).__name__}): {exc}")
        report_service.record_global(
            state,
            "editorial",
            "failed",
            reason=f"{type(exc).__name__}: {exc}",
        )
        logger.info(
            "agent editorial: failed in %.1fs",
            time.perf_counter() - editorial_t0,
        )
    # Flush the terminal status (done or failed) so the dashboard sees
    # the outcome immediately rather than waiting for the puzzles node's
    # entry checkpoint.
    _checkpoint_progress(state)

    return state


# ---------------------------------------------------------------------------
# Node 9: puzzles: themed pool derived from drafts + topic names


async def puzzles(state: AgentState) -> AgentState:
    from app.models import NewspaperConfig

    meta_puz = state.setdefault("meta", {}).setdefault("puzzles", {})
    with Session(engine) as s:
        config = s.get(NewspaperConfig, _uuid(state["config_id"]))
        include_puzzles = bool(config and config.include_puzzles)
        if not include_puzzles:
            state["puzzles_html"] = None
            state["puzzles_meta"] = None
            meta_puz["status"] = "skipped"
            logger.info("agent puzzles: skipped (include_puzzles=False)")
            return state

        topics = s.exec(
            select(Topic).where(Topic.id.in_([_uuid(t) for t in state["topic_ids"]]))
        ).all()
        topic_names = [t.name for t in topics]

    drafts = state.get("drafts", [])
    pool = extract_theme_pool(drafts, topic_names)
    crypto_phrases = extract_crypto_phrases(drafts, state.get("editorial"))

    # Seed is deterministic per issue across processes (hash() is not; it is
    # salted per interpreter by PYTHONHASHSEED). Use sha256 so a re-render in
    # a fresh backend container produces the same puzzles.
    import hashlib

    digest = hashlib.sha256(f"{state['issue_id']}|{state['config_id']}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:4], "big")

    def _run():
        return build_puzzles(
            seed=seed,
            theme_pool=pool,
            wordsearch_pool=pool if pool else None,
            crypto_phrases=crypto_phrases if crypto_phrases else None,
            with_fallback=True,
        )

    try:
        fragment, meta = await asyncio.to_thread(_run)
    except Exception as exc:
        meta_puz["status"] = "failed"
        meta_puz["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
        _warn(state, f"puzzles generation failed: {exc}")
        state["puzzles_html"] = None
        state["puzzles_meta"] = None
        return state

    state["puzzles_html"] = fragment
    state["puzzles_meta"] = meta
    meta_puz["status"] = "fallback" if meta.get("fell_back_to_defaults") else "ok"
    meta_puz["seed"] = meta.get("seed")
    meta_puz["theme_pool_size"] = len(pool)
    meta_puz["crypto_phrase_candidates"] = len(crypto_phrases)
    logger.info(
        "agent puzzles: built (fell_back=%s, givens=%d, fillin=%d words)",
        meta.get("fell_back_to_defaults"),
        meta.get("sudoku_givens"),
        len(meta.get("fillin_words", [])),
    )
    return state


# ---------------------------------------------------------------------------
# Node 10: finalize: persist drafts_json on the issue row


async def finalize(state: AgentState) -> AgentState:
    from app.models import Issue

    drafts = state.get("drafts", [])
    editorial_text = state.get("editorial")

    # Build a flat count funnel from the per-node counters already in meta.
    # This is the shape the UI consumes for the "pipeline trace" panel;
    # individual per-node dicts stay for detail drill-down.
    meta = state.setdefault("meta", {})
    ingest = meta.get("ingest_rss", {}).get("items_new", 0) + meta.get("ingest_search", {}).get(
        "items_added", 0
    )
    meta["funnel"] = {
        "ingested_new": ingest,
        "embedded_candidates": meta.get("embedding", {}).get("total_candidates", 0),
        "after_cross_dedup": meta.get("dedup_breakdown", {}).get("kept", 0),
        "after_select_and_intra_dedup": meta.get("select", {}).get("total_kept", 0),
        "articles_fetched": meta.get("fetch_articles", {}).get("fetched_ok", 0),
        "articles_cached": meta.get("fetch_articles", {}).get("cached", 0),
        "articles_fallback_summary": meta.get("fetch_articles", {}).get("fallback_summary", 0),
        "drafted": meta.get("draft", {}).get("succeeded", len(drafts)),
        "editorial_ok": meta.get("editorial", {}).get("status") == "ok",
        "puzzles_status": meta.get("puzzles", {}).get("status", "skipped"),
    }
    meta.setdefault("run", {})["finished_at"] = now_utc().isoformat()

    for d in drafts:
        report_service.record_event(
            state,
            news_item_id=d.get("news_item_id"),
            step="published",
            status="ok",
            topic_id=d.get("topic_id"),
            topic_name=d.get("topic_name"),
            reason="Included in the rendered issue.",
            reason_source="system",
            data={
                "rendered_title": d.get("rendered_title"),
                "topic_name": d.get("topic_name"),
                "is_lead": bool(d.get("is_lead")),
            },
        )
    report_service.record_global(state, "finalize", "ok", data=meta["funnel"])

    with Session(engine) as s:
        issue = s.get(Issue, _uuid(state["issue_id"]))
        if issue is None:
            _warn(state, "finalize: issue not found")
            return state
        issue.drafts_json = {
            "drafts": drafts,
            "editorial": editorial_text,
            "warnings": state.get("warnings", []),
            "meta": state.get("meta", {}),
            "puzzles_meta": state.get("puzzles_meta"),
            "has_puzzles_html": bool(state.get("puzzles_html")),
        }
        if drafts:
            issue.status = "drafted"
        else:
            issue.status = "failed"
            issue.error = "no drafts produced"
        s.commit()

    logger.info(
        "agent finalize: %d drafts, status=%s", len(drafts), "drafted" if drafts else "failed"
    )
    return state


# ---------------------------------------------------------------------------
# Node 11: render: write HTML to issues/{slug}/, seed story_log embeddings,
# flip issue status to 'ready'.


async def render(state: AgentState) -> AgentState:
    if not state.get("drafts"):
        logger.info("agent render: no drafts, skipping render (status stays 'failed')")
        return state
    try:
        html_path, meta = await render_service.render_issue(_uuid(state["issue_id"]))
        state.setdefault("meta", {})["render"] = {
            "html_path": str(html_path),
            "has_puzzles": meta.get("has_puzzles", False),
        }
        logger.info("agent render: wrote %s", html_path)
    except Exception as exc:
        _warn(state, f"render failed: {exc}")
        report_service.record_global(
            state, "render", "failed", reason=f"{type(exc).__name__}: {exc}"
        )
        # Don't re-raise: finalize already persisted drafts_json, issue stays 'drafted'.
    return state


# ---------------------------------------------------------------------------
# Node 12: notify: email the generated PDF to config.newsletter_emails.
# Runs last so it only fires when the issue is actually 'ready'. A failed
# notify does not flip issue status; the newspaper is already published.


async def notify(state: AgentState) -> AgentState:
    from app.models import Issue

    meta_notify = state.setdefault("meta", {}).setdefault("notify", {})

    with Session(engine) as s:
        config = s.get(NewspaperConfig, _uuid(state["config_id"]))
        issue = s.get(Issue, _uuid(state["issue_id"]))
        recipients = list(config.newsletter_emails) if config else []
        paper_name = config.name if config else "The Daily Crosswire"
        language = config.language if config else "en"
        slug = issue.slug if issue else None
        status_now = issue.status if issue else None
        issue_date = issue.issue_date if issue else None

    if not recipients:
        meta_notify.update({"status": "skipped", "reason": "no subscribers"})
        logger.info("agent notify: skipped (no subscribers)")
        return state

    if status_now != "ready" or not slug:
        meta_notify.update({"status": "skipped", "reason": "issue not ready"})
        logger.info("agent notify: skipped (issue not ready, status=%s)", status_now)
        return state

    if not settings.smtp_configured:
        meta_notify.update({"status": "misconfigured"})
        logger.warning(
            "agent notify: SMTP not configured; skipping send to %d recipient(s)",
            len(recipients),
        )
        return state

    try:
        from app.services import pdf as pdf_service
        from app.services import mail as mail_service

        pdf_bytes = await pdf_service.build_issue_pdf(slug)
    except Exception as exc:
        meta_notify.update({"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
        _warn(state, f"notify PDF generation failed: {type(exc).__name__}: {exc}")
        return state

    payload = mail_service.build_issue_email(
        paper_name=paper_name,
        issue_date=issue_date,
        language=language,
        slug=slug,
        pdf_bytes=pdf_bytes,
        kind="auto",
    )

    try:
        outcomes = await mail_service.send_issue(recipients, payload)
    except Exception as exc:
        meta_notify.update({"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
        _warn(state, f"notify SMTP send failed: {type(exc).__name__}: {exc}")
        return state

    sent = sum(1 for o in outcomes if o.get("status") == "ok")
    failed = sum(1 for o in outcomes if o.get("status") == "failed")
    meta_notify.update(
        {
            "status": "ok" if failed == 0 else "partial",
            "sent": sent,
            "failed": failed,
            "per_recipient": outcomes,
        }
    )
    logger.info("agent notify: sent=%d failed=%d", sent, failed)
    return state


# ---------------------------------------------------------------------------
# Node 13: print: send the generated PDF to a CUPS printer.
# Runs after notify, only when PRINTER is defined in .env. Like notify,
# a print failure does not flip issue status; the newspaper is already published.


async def print_issue(state: AgentState) -> AgentState:
    from app.models import Issue

    meta_print = state.setdefault("meta", {}).setdefault("print", {})

    printer = settings.printer.strip() if settings.printer else ""
    if not printer:
        meta_print.update({"status": "skipped", "reason": "no printer configured"})
        logger.info("agent print: skipped (no printer configured)")
        return state

    with Session(engine) as s:
        issue = s.get(Issue, _uuid(state["issue_id"]))
        slug = issue.slug if issue else None
        status_now = issue.status if issue else None

    if status_now != "ready" or not slug:
        meta_print.update({"status": "skipped", "reason": "issue not ready"})
        logger.info("agent print: skipped (issue not ready, status=%s)", status_now)
        return state

    try:
        from app.services import pdf as pdf_service

        pdf_bytes = await pdf_service.build_issue_pdf(slug)
    except Exception as exc:
        meta_print.update(
            {"status": "failed", "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
        )
        _warn(state, f"print PDF generation failed: {type(exc).__name__}: {exc}")
        return state

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_filename = f"{slug}.pdf"
            pdf_file_path = Path(tmp_dir) / pdf_filename
            pdf_file_path.write_bytes(pdf_bytes)

            cmd = ["lp"]
            if settings.cups_host:
                cmd.extend(["-h", settings.cups_host])
            cmd.extend([
                "-d",
                printer,
                "-o",
                "PageSize=A3",
                "-o",
                "MediaType=PLAIN_NORMAL",
                "-o",
                "Duplex=DuplexNoTumble",
                "-o",
                "InputSlot=Auto",
                pdf_filename,
            ])

            env = {**os.environ}
            if settings.cups_host:
                env["CUPS_SERVER"] = settings.cups_host

            logger.info("agent print: running %s in %s", " ".join(cmd), tmp_dir)
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=tmp_dir,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            err = stderr.decode().strip() or stdout.decode().strip() or f"exit code {proc.returncode}"
            meta_print.update({
                "status": "failed",
                "printer": printer,
                "error": err[:300],
            })
            _warn(state, f"print failed: {err[:200]}")
            logger.error("agent print: lp failed for %s on %s: %s", slug, printer, err)
            return state

        out = stdout.decode().strip()
        meta_print.update({
            "status": "ok",
            "printer": printer,
            "output": out,
        })
        logger.info("agent print: submitted job for %s to %s: %s", slug, printer, out)
    except Exception as exc:
        meta_print.update({
            "status": "failed",
            "printer": printer,
            "error": f"{type(exc).__name__}: {str(exc)[:200]}",
        })
        _warn(state, f"print execution failed: {type(exc).__name__}: {exc}")
        logger.exception("agent print: exception during print for %s", slug)

    return state


print_node = print_issue
print = print_issue
