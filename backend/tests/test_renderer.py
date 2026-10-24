"""Tests for the HTML renderer. Fabricates drafts_json on an Issue row
directly (no agent run), then exercises renderer.render_issue."""

import asyncio
import uuid
from datetime import date

import pytest
from sqlmodel import Session, select

from app.config import settings
from app.db import engine
from app.models import Issue, NewsItem, NewspaperConfig, StoryLog, Topic
from app.services.renderer import render_issue
from app.services.rss import normalize_title

CFG_NAME = "__test_renderer"


def _seed(s: Session) -> dict:
    """Return a dict of primitive IDs, not ORM objects, so tests don't hit
    stale-object issues after render_issue commits in its own session."""
    cfg = NewspaperConfig(
        name=CFG_NAME,
        schedule_kind="interval",
        frequency_days=1,
        run_hour=6,
        include_puzzles=False,  # keep test fast; renderer still writes HTML
    )
    s.add(cfg)
    s.commit()
    s.refresh(cfg)

    topic_a = Topic(config_id=cfg.id, name="Tech", preferred_sources=[])
    topic_b = Topic(config_id=cfg.id, name="Games", preferred_sources=[])
    s.add(topic_a)
    s.add(topic_b)
    s.commit()
    s.refresh(topic_a)
    s.refresh(topic_b)

    items = [
        NewsItem(
            url=f"https://example.com/a-{i}",
            title=f"AI item {i}",
            normalized_title=normalize_title(f"AI item {i}"),
            topic_id=topic_a.id,
        )
        for i in range(2)
    ] + [
        NewsItem(
            url="https://example.com/g-1",
            title="Game item 1",
            normalized_title=normalize_title("Game item 1"),
            topic_id=topic_b.id,
        )
    ]
    for it in items:
        s.add(it)
    s.commit()
    for it in items:
        s.refresh(it)

    issue = Issue(
        config_id=cfg.id,
        slug=f"__test-renderer-{cfg.id}",
        issue_date=date.today(),
        lookback_start=date.today(),
        lookback_end=date.today(),
        status="drafted",
    )
    s.add(issue)
    s.commit()
    s.refresh(issue)

    issue.drafts_json = {
        "drafts": [
            {
                "news_item_id": str(items[0].id),
                "topic_id": str(topic_a.id),
                "topic_name": topic_a.name,
                "rendered_title": "Chipworks ships new reasoning model",
                "body": "Chipworks shipped a new reasoning model today. The mode improves code tasks because the model plans more before acting. For users the result is fewer reruns.",
                "is_lead": True,
                "original_title": items[0].title,
                "original_url": items[0].url,
                "quote": "We are against paying for them twice.",
                "quote_source": "We are against paying for them twice.",
                "quote_attribution": "Ana Ruiz",
                "quote_verified": True,
                "quote_translated": False,
            },
            {
                "news_item_id": str(items[1].id),
                "topic_id": str(topic_a.id),
                "topic_name": topic_a.name,
                "rendered_title": "Open source model hits new retrieval benchmark",
                "body": "A small open model crossed a retrieval benchmark threshold today. The gain came from a new training recipe. For developers this narrows the gap between local and hosted systems.",
                "is_lead": False,
                "original_title": items[1].title,
                "original_url": items[1].url,
                "quote": "Estamos en contra de pagarlas dos veces.",
                "quote_source": "We are against paying for them twice.",
                "quote_attribution": "Ana Ruiz",
                "quote_verified": True,
                "quote_translated": True,
            },
            {
                "news_item_id": str(items[2].id),
                "topic_id": str(topic_b.id),
                "topic_name": topic_b.name,
                "rendered_title": "Harborside unveils holiday console bundle",
                "body": "Harborside announced a holiday console bundle today. The pack adds a new controller and two games because retailers asked for a single SKU. Preorders opened immediately.",
                "is_lead": True,
                "original_title": items[2].title,
                "original_url": items[2].url,
            },
        ],
        "editorial": "Three stories cut across today: the continued push toward models that plan before they answer, the narrowing gap between open and hosted systems, and a quieter hardware story in games that still moves units. The pattern is less about novelty than about compression of what used to take multiple releases.",
        "warnings": [],
        "meta": {},
        "puzzles_meta": None,
        "has_puzzles_html": False,
    }
    s.add(issue)
    s.commit()
    s.refresh(issue)

    return {
        "issue_id": issue.id,
        "issue_slug": issue.slug,
        "topic_a_id": topic_a.id,
        "topic_b_id": topic_b.id,
        "item_ids": [i.id for i in items],
    }


@pytest.fixture
def seeded(clean_db):
    with Session(engine) as s:
        seed = _seed(s)
    # Session is closed before yielding, so tests get clean IDs and must
    # open their own sessions to read state. Prevents stale-object errors
    # after render_issue commits in its own session.
    yield seed


def test_render_writes_files_and_marks_ready(seeded):
    issue_id = seeded["issue_id"]
    issue_slug = seeded["issue_slug"]

    html_path, meta = asyncio.run(render_issue(issue_id))

    assert html_path.exists()
    assert html_path.name == "index.html"

    style_css = html_path.parent / "style.css"
    assert style_css.exists(), "style.css should have been copied next to index.html"

    html = html_path.read_text()
    assert "The Daily Crosswire" in html
    assert "Chipworks ships new reasoning model" in html  # lead of topic A
    assert "Harborside unveils holiday console bundle" in html  # lead of topic B
    assert "Open source model hits new retrieval benchmark" in html  # second story
    assert "Tech" in html and "Games" in html  # beat headings
    assert "Editorial" in html
    # Pull quotes render for leads (full style) and secondary stories
    # (compact style), with the translated text and the attribution.
    assert 'class="pullquote"' in html
    assert 'class="pullquote pullquote-compact"' in html
    assert "We are against paying for them twice." in html
    assert "Estamos en contra de pagarlas dos veces." in html
    assert "Ana Ruiz" in html
    # Games markers only render when puzzles_html is set; this fixture
    # has no puzzles, so assert the negative.
    assert "GAMES_PAGE_START" not in html

    with Session(engine) as s:
        reloaded = s.get(Issue, issue_id)
        assert reloaded.status == "ready"
        assert reloaded.html_path == f"issues/{issue_slug}/index.html"

        logs = s.exec(select(StoryLog).where(StoryLog.issue_id == issue_id)).all()
        assert len(logs) == 3
        titles = {log.rendered_title for log in logs}
        assert "Chipworks ships new reasoning model" in titles


def test_render_is_idempotent(seeded):
    issue_id = seeded["issue_id"]

    path1, _ = asyncio.run(render_issue(issue_id))
    path2, _ = asyncio.run(render_issue(issue_id))
    assert path1 == path2

    with Session(engine) as s:
        logs = s.exec(select(StoryLog).where(StoryLog.issue_id == issue_id)).all()
        # Re-rendering should not duplicate story_log rows.
        assert len(logs) == 3
