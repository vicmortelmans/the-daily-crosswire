"""Takes a drafted Issue (status='drafted') and produces the final
issues/{slug}/index.html + style.css. Writes story_log rows with
title embeddings so Phase 7 dedup has something to match against next run."""

import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlmodel import Session, select

from app.config import settings
from app.db import engine
from app.models import Issue, NewsItem, NewspaperConfig, StoryLog, Topic
from app.services import embeddings as emb_service
from app.utils.time import now_utc

logger = logging.getLogger(__name__)

PAPER_NAME = "The Daily Crosswire"
TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"

_env = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
    trim_blocks=True,
    lstrip_blocks=True,
)


def _domain(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        return urlparse(url).netloc or None
    except Exception:
        return None


def _group_beats(drafts: list[dict], topics: list[Topic]) -> list[dict]:
    """Bucket drafts by topic_id, set lead, preserve topic order."""
    topics_by_id = {str(t.id): t for t in topics}
    by_topic: dict[str, list[dict]] = {}
    for d in drafts:
        by_topic.setdefault(d["topic_id"], []).append(d)

    beats: list[dict] = []
    # Preserve topic creation order from `topics` arg
    for t in topics:
        tid = str(t.id)
        bucket = by_topic.get(tid, [])
        if not bucket:
            continue
        lead: Optional[dict] = None
        others: list[dict] = []
        for d in bucket:
            enriched = {
                **d,
                "source_domain": _domain(d.get("original_url")),
            }
            if d.get("is_lead") and lead is None:
                lead = enriched
            else:
                others.append(enriched)
        # Fallback: if no lead was flagged, promote first
        if lead is None and others:
            lead = others.pop(0)
        beats.append(
            {
                "topic_name": t.name,
                "topic_id": tid,
                "lead": lead,
                "stories": others,
            }
        )
    return beats


def _pick_hero(beats: list[dict]) -> Optional[dict]:
    """Hero = lead of the topic with the most stories (ties broken by
    topic order)."""
    if not beats:
        return None
    ranked = sorted(
        beats,
        key=lambda b: (1 + len(b["stories"])) if b.get("lead") else len(b["stories"]),
        reverse=True,
    )
    for b in ranked:
        if b.get("lead"):
            return {
                **b["lead"],
                "topic_name": b["topic_name"],
                "topic_id": b["topic_id"],
            }
    return None


def _build_sources(drafts: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for d in drafts:
        url = d.get("original_url")
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(
            {
                "url": url,
                "title": d.get("original_title") or d.get("rendered_title") or url,
                "domain": _domain(url),
            }
        )
    return out


async def render_issue(issue_id: uuid.UUID) -> tuple[Path, dict[str, Any]]:
    """Render an issue to disk. Returns (html_path, meta_dict).

    Side effects:
      - Writes issues/{slug}/index.html, issues/{slug}/style.css, issues/{slug}/meta.json
      - Inserts StoryLog rows (one per draft) with rendered_title_embedding
        (best-effort; if the embeddings endpoint is down, stores NULL).
      - Updates Issue row: html_path, status='ready'.
    """
    import json

    with Session(engine) as s:
        issue = s.get(Issue, issue_id)
        if issue is None:
            raise ValueError(f"issue {issue_id} not found")
        if not issue.drafts_json:
            raise ValueError(f"issue {issue_id} has no drafts_json (was the agent run?)")

        config = s.get(NewspaperConfig, issue.config_id)
        topics = s.exec(select(Topic).where(Topic.config_id == issue.config_id)).all()

        drafts = issue.drafts_json.get("drafts", [])
        editorial_text = issue.drafts_json.get("editorial")
        puzzles_html = None
        # Puzzles HTML isn't persisted in drafts_json (too big). We regenerate
        # if the meta says there were puzzles and we have drafts to pull themes
        # from; or we skip. For the MVP path we read from the state produced
        # by the agent and passed to this function via drafts_json's puzzles_html
        # key if ever set, else fall back to building from the stored meta.
        puzzles_meta = issue.drafts_json.get("puzzles_meta")

        beats = _group_beats(drafts, topics)
        hero = _pick_hero(beats)
        # Reorder beats so the hero's topic renders directly after the hero
        # block. Keeps the rest in original topic order.
        if hero and hero.get("topic_id") and beats:
            hero_tid = hero["topic_id"]
            beats = [b for b in beats if b["topic_id"] == hero_tid] + [
                b for b in beats if b["topic_id"] != hero_tid
            ]
        sources = _build_sources(drafts)

        generated_at = now_utc()
        # Simple volume/number: volume = year, number = day-of-year
        vol = issue.issue_date.year if issue.issue_date else generated_at.year
        num = (
            issue.issue_date.timetuple().tm_yday
            if issue.issue_date
            else generated_at.timetuple().tm_yday
        )

        # Rebuild puzzles if needed (deterministic from the same seed the agent used)
        if config and config.include_puzzles and puzzles_meta:
            try:
                import hashlib
                from app.puzzles.build import build_puzzles
                from app.puzzles.nouns import (
                    extract_crypto_phrases,
                    extract_theme_pool,
                )

                pool = extract_theme_pool(drafts, [t.name for t in topics])
                crypto_phrases = extract_crypto_phrases(drafts, editorial_text)
                seed = puzzles_meta.get("seed")
                if seed is None:
                    digest = hashlib.sha256(
                        f"{issue.id}|{issue.config_id}".encode("utf-8")
                    ).digest()
                    seed = int.from_bytes(digest[:4], "big")
                puzzles_html, _ = build_puzzles(
                    seed=seed,
                    theme_pool=pool,
                    wordsearch_pool=pool if pool else None,
                    crypto_phrases=crypto_phrases if crypto_phrases else None,
                    issue_date=issue.issue_date,
                    with_fallback=True,
                )
            except Exception:
                logger.exception("renderer: puzzles rebuild failed, shipping without games page")
                puzzles_html = None

        from app.agent.languages import (
            format_long_date,
            format_short_time,
            template_strings,
        )

        lang_code = getattr(config, "language", "en") if config else "en"
        strings = template_strings(lang_code)

        date_for_fmt = issue.issue_date or generated_at
        context = {
            "paper_name": config.name if config else "The Daily Crosswire"

            "issue_date_pretty": format_long_date(date_for_fmt, lang_code),
            "issue_id_short": str(issue.id).split("-")[0],
            "generated_at_pretty": generated_at.strftime("%Y-%m-%d %H:%M UTC"),
            "generated_time_short": format_short_time(generated_at, lang_code),
            "lookback_start": issue.lookback_start.isoformat() if issue.lookback_start else "-",
            "lookback_end": issue.lookback_end.isoformat() if issue.lookback_end else "-",
            "topic_list": ", ".join(t.name for t in topics),
            "volume": vol,
            "number": num,
            "hero": hero,
            "beats": beats,
            "editorial": editorial_text,
            "sources": sources,
            "puzzles_html": puzzles_html,
            "language": lang_code,
            "strings": strings,
        }

        tpl = _env.get_template("issue.html.j2")
        html = tpl.render(**context)

        # Write to disk
        issue_dir = settings.issues_dir / issue.slug
        issue_dir.mkdir(parents=True, exist_ok=True)
        html_path = issue_dir / "index.html"
        html_path.write_text(html, encoding="utf-8")

        # Copy style.css from the backend's own templates dir. Kept inside
        # the package so the backend image is self-contained (no host
        # mount required).
        src_css = TEMPLATE_DIR / "style.css"
        if src_css.exists():
            shutil.copy2(src_css, issue_dir / "style.css")
        else:
            logger.warning("style.css not found at %s; issue will be unstyled", src_css)

        meta_json = {
            "issue_id": str(issue.id),
            "slug": issue.slug,
            "issue_date": issue.issue_date.isoformat() if issue.issue_date else None,
            "topics": [t.name for t in topics],
            "num_drafts": len(drafts),
            "has_editorial": bool(editorial_text),
            "has_puzzles": bool(puzzles_html),
            "generated_at": generated_at.isoformat(),
        }
        (issue_dir / "meta.json").write_text(json.dumps(meta_json, indent=2), encoding="utf-8")

        # Embed outside the transaction so a slow embedder doesn't hold locks.
        titles_to_embed = [d.get("rendered_title", "") for d in drafts]
        embeddings: list[Optional[list[float]]] = [None] * len(drafts)
        if titles_to_embed:
            try:
                vecs = await emb_service.embed(titles_to_embed)
                if len(vecs) == len(titles_to_embed):
                    embeddings = vecs  # type: ignore[assignment]
            except Exception:
                logger.exception(
                    "renderer: embedding rendered_titles failed; story_log rows "
                    "will be written without embeddings"
                )

        # Single atomic block: delete old story_log + insert new + flip status.
        # If any step raises, the whole thing rolls back and status stays
        # 'drafted' so the run can be re-rendered later without orphan rows.
        for old in s.exec(select(StoryLog).where(StoryLog.issue_id == issue.id)).all():
            s.delete(old)
        for d, emb in zip(drafts, embeddings):
            s.add(
                StoryLog(
                    issue_id=issue.id,
                    news_item_id=uuid.UUID(d["news_item_id"]),
                    topic_id=uuid.UUID(d["topic_id"]) if d.get("topic_id") else None,
                    rendered_title=d["rendered_title"],
                    rendered_body=d.get("body"),
                    rendered_title_embedding=emb,
                )
            )
        issue.html_path = f"issues/{issue.slug}/index.html"
        issue.status = "ready"
        s.add(issue)
        s.commit()

    logger.info("renderer: wrote %s", html_path)
    return html_path, meta_json
