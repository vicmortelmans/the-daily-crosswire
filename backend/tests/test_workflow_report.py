"""Unit tests for the optional per-article workflow report.

Pure in-memory + filesystem tests; no database or LLM calls.
"""

from __future__ import annotations

import json
import types

from app.services import workflow_report
from app.services.workflow_report import (
    group_articles,
    init_report,
    record_event,
    record_retrieved,
    write_report,
)


def _state(enabled: bool = True) -> dict:
    state: dict = {"meta": {}, "warnings": []}
    init_report(
        state,
        enabled=enabled,
        issue_id="11111111-1111-1111-1111-111111111111",
        slug="__test-report",
        issue_date="2026-10-23",
        config_name="Test Paper",
        language="en",
        lookback_start="2026-10-16",
        lookback_end="2026-10-23",
    )
    return state


def _item(nid: str, title: str = "A story") -> dict:
    return {
        "id": nid,
        "title": title,
        "url": f"https://example.com/{nid}",
        "source_domain": "example.com",
        "summary": "Teaser <script>alert(1)</script>",
        "published_at": None,
        "ingested_via": "rss",
        "source_feed_url": "https://example.com/feed.xml",
        "source_query": None,
        "topic_id": "22222222-2222-2222-2222-222222222222",
    }


def test_group_articles_preserves_first_seen_order_and_outcomes():
    state = _state()
    record_retrieved(state, _item("a", "First"), topic_name="Faith", via="rss")
    record_retrieved(state, _item("b", "Second"), topic_name="Faith", via="rss")
    record_event(
        state,
        news_item_id="a",
        step="select",
        status="dropped",
        reason="promo card",
        reason_source="ai",
    )
    record_event(
        state,
        news_item_id="b",
        step="draft",
        status="ok",
        data={"rendered_title": "Second", "body": "Body."},
    )
    record_event(
        state,
        news_item_id="b",
        step="published",
        status="ok",
        data={"topic_name": "Faith"},
    )

    grouped = group_articles(state["workflow_report"])

    assert [a["article"]["news_item_id"] for a in grouped] == ["a", "b"]
    assert grouped[0]["outcome"]["kind"] == "dropped"
    assert grouped[0]["outcome"]["stage"] == "select"
    assert grouped[1]["outcome"]["kind"] == "published"
    # Dropped article gets explicit "not reached" placeholders after select.
    steps = [s["step"] for s in grouped[0]["steps"]]
    assert "select" in steps and "draft" in steps
    assert grouped[0]["steps"][-1]["status"] == "not_reached"


def test_draft_failure_is_terminal():
    state = _state()
    record_retrieved(state, _item("a"), topic_name="Faith", via="rss")
    record_event(
        state,
        news_item_id="a",
        step="draft",
        status="failed",
        reason="timeout",
    )
    grouped = group_articles(state["workflow_report"])
    assert grouped[0]["outcome"]["kind"] == "draft_failed"


def test_write_report_creates_artifacts_and_escapes_html(tmp_path, monkeypatch):
    monkeypatch.setattr(
        workflow_report, "settings", types.SimpleNamespace(issues_dir=tmp_path)
    )
    state = _state()
    record_retrieved(state, _item("a", "Title <b>bold</b>"), topic_name="Faith", via="rss")
    record_event(
        state,
        news_item_id="a",
        step="select",
        status="dropped",
        reason="promo <script>alert(1)</script>",
        reason_source="ai",
    )

    path = write_report(state, final=True)

    assert path is not None and path.exists()
    html = path.read_text(encoding="utf-8")
    assert "Title &lt;b&gt;bold&lt;/b&gt;" in html
    assert "<script>alert(1)</script>" not in html
    assert "promo &lt;script&gt;" in html

    ledger = tmp_path / "__test-report" / "workflow-report.json"
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    assert payload["issue"]["slug"] == "__test-report"
    assert payload["run"]["final"] is True
    assert len(payload["events"]) == 2

    meta = json.loads(
        (tmp_path / "__test-report" / "workflow-report.meta.json").read_text(encoding="utf-8")
    )
    assert meta["final"] is True
    assert meta["article_count"] == 1


def test_disabled_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        workflow_report, "settings", types.SimpleNamespace(issues_dir=tmp_path)
    )
    state = _state(enabled=False)
    record_retrieved(state, _item("a"), topic_name="Faith", via="rss")
    assert state["workflow_report"]["events"] == []
    assert write_report(state, final=True) is None
    assert not (tmp_path / "__test-report").exists()
