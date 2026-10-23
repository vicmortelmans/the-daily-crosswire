import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from app.config import settings
from app.services.article_extractor import (
    extract_text_from_html,
    fetch_article_text,
    MIN_WORD_COUNT,
)


def test_extract_text_clean_article():
    html = """
    <!DOCTYPE html>
    <html>
    <head><title>Test News Article</title></head>
    <body>
      <header><nav><a href="/">Home</a><a href="/politics">Politics</a></nav></header>
      <main>
        <article>
          <h1>City Council Approves New Solar Initiative</h1>
          <p>The city council voted unanimously on Tuesday night to approve an ambitious municipal solar power initiative aimed at reducing citywide emissions by thirty percent over the next decade.</p>
          <p>Under the newly enacted program, all public municipal buildings including city hall, police substations, and public libraries will receive rooftop solar panel installations beginning early next spring.</p>
          <p>Civic leaders and environmental organizations praised the bipartisan agreement, noting that the measure will lower public utility expenditures while demonstrating actionable commitment to sustainable infrastructure.</p>
        </article>
      </main>
      <footer>
        <p>Copyright 2026 Daily Crosswire News. All rights reserved.</p>
      </footer>
    </body>
    </html>
    """
    text = extract_text_from_html(html)
    assert text is not None
    assert "City Council Approves" in text or "city council voted" in text
    assert "Copyright" not in text
    assert "Politics" not in text
    assert len(text.split()) >= MIN_WORD_COUNT


def test_extract_text_truncates_long_body():
    """Long-read/liveblog bodies are capped before they reach the DB and the
    select/draft prompts."""
    paragraph = (
        "The city council met on Tuesday to discuss the new budget and approved "
        "additional funding for road repairs across the district"
    )
    html = (
        "<html><body><article>"
        + "".join(f"<p>{paragraph}, section {i}.</p>" for i in range(80))
        + "</article></body></html>"
    )

    text = extract_text_from_html(html)
    assert text is not None
    assert len(text) <= settings.article_max_characters
    assert len(text) > MIN_WORD_COUNT
    assert text.startswith("The city council met on Tuesday")
    # The cut must actually happen: the source is ~10k chars, so the result
    # should sit right at the cap (allow a few chars for `.rstrip()`).
    assert len(text) >= settings.article_max_characters - 100


def test_extract_text_too_short_or_empty():
    # Empty string
    assert extract_text_from_html("") is None
    assert extract_text_from_html("   ") is None

    # Cookie notice or 1-sentence gate
    cookie_wall = "<html><body><p>Please accept all cookies to continue reading.</p></body></html>"
    assert extract_text_from_html(cookie_wall) is None


@pytest.mark.asyncio
async def test_fetch_article_text_mocked_success():
    fake_html = """
    <html><body><article>
      <h1>Major Tech Merger Finalized</h1>
      <p>European regulatory authorities officially authorized the landmark acquisition today after protracted antitrust evaluations spanning nine consecutive months of intensive economic scrutiny.</p>
      <p>The multinational conglomerate confirmed that operational integration will commence immediately, with divisional headquarters remaining centered in Amsterdam to maintain continuity of European operations.</p>
      <p>Industry analysts forecast minimal consumer disruption, though competing telecommunications firms have expressed reservations concerning market concentration across core northern European broadband infrastructure segments.</p>
    </article></body></html>
    """
    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.headers = {"content-type": "text/html; charset=utf-8"}
    mock_resp.text = fake_html

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_resp

    text = await fetch_article_text("https://example.com/story-1", client=mock_client)
    assert text is not None
    assert "Major Tech Merger Finalized" in text or "European regulatory authorities" in text


@pytest.mark.asyncio
async def test_fetch_article_text_mocked_http_error():
    mock_resp = AsyncMock()
    mock_resp.status_code = 403
    mock_resp.headers = {"content-type": "text/html"}

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_resp

    # Should return None instead of raising, triggering graceful fallback
    text = await fetch_article_text("https://example.com/paywalled", client=mock_client)
    assert text is None

