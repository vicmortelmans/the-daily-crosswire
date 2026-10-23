import logging
import re
from datetime import datetime
from urllib.parse import urlparse

import feedparser
from sqlmodel import Session, select

from app.db import engine
from app.models import NewsItem, RSSFeed
from app.utils.time import now_utc

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (compatible; newspaper-agent/0.1; +https://localhost)"

_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")
_HTML_RE = re.compile(r"<[^>]+>")


def normalize_title(title: str) -> str:
    t = title.lower().strip()
    t = _PUNCT_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t)
    return t.strip()


def _parse_published(entry) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        value = entry.get(key)
        if value:
            try:
                return datetime(*value[:6])
            except (TypeError, ValueError):
                continue
    return None


def _clean_summary(raw: str | None) -> str | None:
    if not raw:
        return None
    cleaned = _HTML_RE.sub("", raw).strip()
    return cleaned[:2000] if cleaned else None


def poll_feed(session: Session, feed: RSSFeed) -> list[NewsItem]:
    """Poll a single RSS feed, upsert new NewsItems, return the inserted rows."""
    parsed = feedparser.parse(
        feed.url,
        etag=feed.last_etag or None,
        modified=feed.last_modified or None,
        agent=USER_AGENT,
    )

    bozo_exc = parsed.get("bozo_exception")
    if parsed.get("bozo") and bozo_exc and not parsed.entries:
        logger.warning("feedparser bozo for %s: %s", feed.url, bozo_exc)

    added_items: list[NewsItem] = []
    for entry in parsed.entries:
        url = entry.get("link")
        title = entry.get("title")
        if not url or not title:
            continue

        existing = session.exec(select(NewsItem).where(NewsItem.url == url)).first()
        if existing is not None:
            continue

        item = NewsItem(
            url=url,
            title=title,
            normalized_title=normalize_title(title),
            summary=_clean_summary(entry.get("summary") or entry.get("description")),
            source_domain=urlparse(url).netloc or None,
            published_at=_parse_published(entry),
            topic_id=feed.topic_id,
            ingested_via="rss",
            source_feed_url=feed.url,
        )
        session.add(item)
        added_items.append(item)

    feed.last_polled_at = now_utc()
    feed.last_etag = parsed.get("etag") or feed.last_etag
    feed.last_modified = parsed.get("modified") or feed.last_modified

    session.commit()
    return added_items


def poll_all_feeds() -> dict:
    """Invoked by APScheduler every hour. Opens its own session."""
    results: dict[str, int | str] = {}
    with Session(engine) as session:
        feeds = session.exec(select(RSSFeed)).all()
        for feed in feeds:
            try:
                items = poll_feed(session, feed)
                results[str(feed.id)] = len(items)
                logger.info("polled %s -> %d new", feed.url, len(items))
            except Exception as exc:
                session.rollback()
                logger.exception("failed to poll %s", feed.url)
                results[str(feed.id)] = f"error: {exc}"
    return results
