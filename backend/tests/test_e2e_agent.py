"""End-to-end tests for the agent pipeline.

Run: docker compose exec backend pytest tests/ -v

Uses the real stack: RSS (hnrss.org), embeddings (ollama via EMBEDDING_*
env), and a real chat LLM (OPENAI_API_KEY). No mocks.
"""

import uuid
from datetime import date, timedelta

from sqlmodel import Session, select

from app.db import engine
from app.models import Issue, NewsItem, StoryLog, Topic

CFG_NAME = "__test_e2e_agent"


def _create_config_with_feed(client):
    cfg = client.post(
        "/api/configs",
        json={"name": CFG_NAME, "schedule_kind": "interval", "frequency_days": 1},
    ).json()
    topic = client.post(f"/api/configs/{cfg['id']}/topics", json={"name": "AI/LLMs"}).json()
    client.post(
        f"/api/topics/{topic['id']}/feeds",
        json={"url": "https://hnrss.org/newest?q=LLM&count=10"},
    )
    return cfg["id"], topic["id"]


def test_agent_end_to_end(client, clean_db):
    """Full pipeline run. Agent may keep 0 or more stories (LLM's call).
    Asserts the graph completes, RSS ingest worked, and any drafts
    produced follow voice rules."""
    cfg_id, _ = _create_config_with_feed(client)

    resp = client.post("/api/debug/agent", json={"config_id": cfg_id})
    assert resp.status_code == 200, resp.text
    result = resp.json()

    assert result["issue_id"]
    assert result["candidates"] > 0, "RSS ingest produced zero news_items"
    # `ready` if render succeeded, `drafted` if render failed gracefully,
    # `failed` if the LLM dropped every candidate.
    assert result["status"] in ("ready", "drafted", "failed")

    with Session(engine) as s:
        issue = s.get(Issue, uuid.UUID(result["issue_id"]))
        assert issue is not None
        assert issue.drafts_json is not None  # always populated, even on failed

        for d in issue.drafts_json.get("drafts", []):
            assert d.get("rendered_title")
            assert d.get("body")
            assert "!" not in d["body"], f"exclamation leaked into draft: {d['rendered_title']}"
            if d.get("quote"):
                # Every printed quote must trace back to a literal span of the
                # fetched article; the translation is derived from that span.
                assert d.get("quote_verified") is True
                assert d.get("quote_source")
                item = s.get(NewsItem, uuid.UUID(d["news_item_id"]))
                assert item is not None and item.article_text
                from app.agent.nodes import _normalize_for_match

                assert _normalize_for_match(d["quote_source"]) in _normalize_for_match(
                    item.article_text
                ), f"pull quote is not a verbatim span: {d['rendered_title']}"

        if issue.drafts_json.get("drafts"):
            leads = [d for d in issue.drafts_json["drafts"] if d.get("is_lead")]
            assert len(leads) >= 1

        # If rendered, files must exist on disk and be reachable via static mount.
        if issue.status == "ready":
            assert issue.html_path
            from pathlib import Path

            html_file = Path("/app") / issue.html_path
            assert html_file.exists(), f"rendered html missing at {html_file}"


def test_dedup_drops_stories_already_in_story_log(client, clean_db):
    """Directly seed a NewsItem + StoryLog under the test's topic (no RSS
    poll; that can silently skip when another config owns the same URL),
    then run the agent and assert dedup flagged the seeded URL.

    The test creates its own feed URL with a unique marker so even if the
    user's DB has overlapping feeds, there's no collision on this URL.
    """
    from app.services.rss import normalize_title

    cfg_id, topic_id_str = _create_config_with_feed(client)

    # Synthesize a news_item tied to this test topic, plus a past Issue +
    # StoryLog referencing it. That's all dedup needs to fire.
    past_url = f"https://example.invalid/__test_dedup/{uuid.uuid4()}"
    past_title = "Anthropic ships Claude Opus 5 with 2M context window"

    with Session(engine) as s:
        topic = s.get(Topic, uuid.UUID(topic_id_str))
        past_item = NewsItem(
            url=past_url,
            title=past_title,
            normalized_title=normalize_title(past_title),
            topic_id=topic.id,
        )
        s.add(past_item)
        s.commit()
        s.refresh(past_item)

        past_issue = Issue(
            config_id=uuid.UUID(cfg_id),
            slug=f"__test_seed_past_{uuid.uuid4().hex[:8]}",
            issue_date=date.today() - timedelta(days=1),
            status="ready",
        )
        s.add(past_issue)
        s.commit()
        s.refresh(past_issue)

        s.add(
            StoryLog(
                issue_id=past_issue.id,
                news_item_id=past_item.id,
                topic_id=topic.id,
                rendered_title=past_title,
            )
        )
        s.commit()

    resp = client.post("/api/debug/agent", json={"config_id": cfg_id}).json()
    breakdown = resp.get("meta", {}).get("dedup_breakdown", {})
    assert (
        breakdown.get("dropped_url", 0) >= 1
    ), f"expected dropped_url >= 1 for seeded URL {past_url}; got breakdown={breakdown}"


def test_agent_searxng_path_for_feedless_topic(client, clean_db):
    """A topic with no RSS feeds should fall through to SearXNG in
    `ingest_search`. Assert news_items get created via that path."""
    cfg = client.post(
        "/api/configs",
        json={"name": CFG_NAME, "schedule_kind": "interval", "frequency_days": 7},
    ).json()
    topic = client.post(
        f"/api/configs/{cfg['id']}/topics",
        json={
            "name": "artificial intelligence",
            "search_hints": "recent news releases",
        },
    ).json()
    # No feeds intentionally.

    resp = client.post("/api/debug/agent", json={"config_id": cfg["id"]})
    assert resp.status_code == 200, resp.text
    result = resp.json()

    # ingest_search must have produced candidates.
    assert result["candidates"] > 0, (
        f"ingest_search produced no news_items for topic without feeds. "
        f"warnings={result.get('warnings', [])}"
    )

    # And those news_items should be tied to this topic.
    with Session(engine) as s:
        items = s.exec(select(NewsItem).where(NewsItem.topic_id == uuid.UUID(topic["id"]))).all()
        assert len(items) >= 1, "no news_items tied to the feedless topic"
        # SearXNG results should have real URLs and titles.
        assert all(i.url and i.title for i in items)
