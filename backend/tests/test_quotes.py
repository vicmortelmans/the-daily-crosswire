"""Tests for the pull-quote picker.

Run: docker compose exec backend pytest tests/test_quotes.py -v

The helper tests are pure Python. The node tests need the stack (Postgres via
`clean_db`) but stub the LLM entirely, so they never call a model.
"""

import asyncio
import uuid

from sqlmodel import Session

from app.agent import nodes
from app.db import engine
from app.models import NewsItem, NewspaperConfig, Topic
from app.services.rss import normalize_title

ARTICLE = (
    "The city council voted on Tuesday to end the street-cleaning pilot. "
    "\u201cWe are not against cleaner streets,\u201d said council member Ana Ruiz. "
    "\u201cWe are against paying for them twice.\u201d "
    "The program cost 12 million dollars over two years and served four districts. "
    "Officials said a replacement plan would be presented in March."
)

SOURCE_QUOTE = "We are against paying for them twice."
TRANSLATED_QUOTE = "Estamos en contra de pagarlas dos veces."


# ---------------------------------------------------------------------------
# Pure helpers


def test_verify_quote_accepts_typography_and_whitespace_variants():
    # Straight quotes, extra spaces, newlines: still the same text.
    assert nodes._verify_quote(
        '  "We are against paying\n for them twice."  ',
        ARTICLE,
    )
    assert nodes._verify_quote(SOURCE_QUOTE.lower(), ARTICLE.upper())


def test_verify_quote_rejects_paraphrase():
    assert not nodes._verify_quote(
        "We object to paying for street cleaning more than once.", ARTICLE
    )


def test_verify_quote_rejects_empty_and_missing_text():
    assert not nodes._verify_quote("", ARTICLE)
    assert not nodes._verify_quote(SOURCE_QUOTE, None)


def test_normalize_for_match_collapses_typographic_variants():
    assert nodes._normalize_for_match("\u201cCaf\u00e9 \u2014 10\u00a0000\u201d") == (
        '"caf\u00e9 - 10 000"'
    )


def test_translation_ok_requires_all_digits():
    source = "The program cost 12 million dollars over two years."
    assert nodes._translation_ok(source, "El programa cost\u00f3 12 millones de d\u00f3lares.")
    assert not nodes._translation_ok(source, "El programa cost\u00f3 millones de d\u00f3lares.")


def test_translation_ok_rejects_empty_and_runaway_output():
    source = "We are against paying for them twice."
    assert not nodes._translation_ok(source, "   ")
    assert not nodes._translation_ok(source, "palabra " * 200)


def test_format_safe_keeps_braces_in_article_text():
    out = nodes._format_safe(
        "title={title} body={article_text}",
        title="T",
        article_text="a {brace} and {another}",
    )
    assert out == "title=T body=a {brace} and {another}"


# ---------------------------------------------------------------------------
# Node tests (LLM stubbed)


def _seed(
    article_text: str = ARTICLE,
    title: str = "Council ends street-cleaning pilot",
) -> dict:
    suffix = uuid.uuid4().hex[:8]
    with Session(engine) as s:
        cfg = NewspaperConfig(
            name=f"__test-quotes-{suffix}",
            schedule_kind="interval",
            frequency_days=1,
            include_puzzles=False,
        )
        s.add(cfg)
        s.commit()
        s.refresh(cfg)
        topic = Topic(config_id=cfg.id, name="City")
        s.add(topic)
        s.commit()
        s.refresh(topic)
        item = NewsItem(
            url=f"https://example.com/quote-{suffix}",
            title=title,
            normalized_title=normalize_title(title),
            summary=article_text[:200],
            article_text=article_text,
            topic_id=topic.id,
        )
        s.add(item)
        s.commit()
        s.refresh(item)
        return {
            "config_id": str(cfg.id),
            "topic_id": str(topic.id),
            "item_id": str(item.id),
        }


def _state(seed: dict) -> dict:
    return {
        "config_id": seed["config_id"],
        "topic_ids": [seed["topic_id"]],
        "selected_by_topic": {seed["topic_id"]: [seed["item_id"]]},
        "meta": {},
        "warnings": [],
    }


def _patch_llm(monkeypatch, pick_outputs, translate_output=None):
    """Stub the structured LLMs. `pick_outputs` is consumed in order (first
    pick, then repair); an Exception instance in the list is raised instead of
    returned. `translate_output` may likewise be an Exception."""
    assert isinstance(pick_outputs, list)
    calls = {"pick": 0, "translate": 0}

    async def _invoke(llm, messages, attempts=3, *, state=None, stage=None):
        if stage == "quote":
            out = pick_outputs[min(calls["pick"], len(pick_outputs) - 1)]
            calls["pick"] += 1
            if isinstance(out, Exception):
                raise out
            return out
        if stage == "quote_translate":
            calls["translate"] += 1
            if isinstance(translate_output, Exception):
                raise translate_output
            return translate_output
        raise AssertionError(f"unexpected stage {stage!r}")

    monkeypatch.setattr(nodes, "chat_llm_configured", lambda: True)
    monkeypatch.setattr(nodes, "get_structured_llm", lambda *a, **k: object())
    monkeypatch.setattr(nodes, "_invoke_with_parse_retry", _invoke)
    return calls


def _pick(**kwargs) -> nodes._QuotePickOutput:
    return nodes._QuotePickOutput(**kwargs)


def _translation(**kwargs) -> nodes._QuoteTranslateOutput:
    return nodes._QuoteTranslateOutput(**kwargs)


def test_quote_picks_verifies_and_translates(monkeypatch, clean_db):
    seed = _seed()
    _patch_llm(
        monkeypatch,
        [_pick(quote=SOURCE_QUOTE, attribution="Ana Ruiz", reason="Plain-spoken.")],
        _translation(quote=TRANSLATED_QUOTE, attribution="Ana Ruiz"),
    )

    state = _state(seed)
    asyncio.run(nodes.quote(state))

    stored = state["quotes"][seed["item_id"]]
    assert stored["quote"] == TRANSLATED_QUOTE
    assert stored["quote_source"] == SOURCE_QUOTE
    assert stored["quote_attribution"] == "Ana Ruiz"
    assert stored["quote_verified"] is True
    assert stored["quote_translated"] is True
    assert state["meta"]["quote"]["quoted"] == 1
    assert state["meta"]["quote"]["verified"] == 1
    assert state["meta"]["quote"]["translated"] == 1


def test_quote_repairs_a_paraphrased_pick(monkeypatch, clean_db):
    seed = _seed()
    calls = _patch_llm(
        monkeypatch,
        [
            _pick(quote="We object to paying twice for street cleaning."),
            _pick(quote=SOURCE_QUOTE, attribution="Ana Ruiz"),
        ],
        _translation(quote=TRANSLATED_QUOTE),
    )

    state = _state(seed)
    asyncio.run(nodes.quote(state))

    assert calls["pick"] == 2
    assert state["quotes"][seed["item_id"]]["quote_source"] == SOURCE_QUOTE
    assert state["meta"]["quote"]["unverified"] == 0


def test_quote_drops_unverifiable_pick(monkeypatch, clean_db):
    seed = _seed()
    _patch_llm(
        monkeypatch,
        [_pick(quote="We object to paying twice for street cleaning.")],
        _translation(quote="no deberia usarse"),
    )

    state = _state(seed)
    asyncio.run(nodes.quote(state))

    assert state["quotes"] == {}
    assert state["meta"]["quote"]["unverified"] == 1
    assert state["meta"]["quote"]["quoted"] == 0


def test_quote_respects_null_pick(monkeypatch, clean_db):
    seed = _seed()
    _patch_llm(monkeypatch, [_pick(quote=None, reason="Only routine detail.")])

    state = _state(seed)
    asyncio.run(nodes.quote(state))

    assert state["quotes"] == {}
    assert state["meta"]["quote"]["no_quote"] == 1


def test_quote_rejects_the_headline(monkeypatch, clean_db):
    headline = "Council votes to end street cleaning pilot program"
    seed = _seed(article_text=f"{headline}. " + ARTICLE, title=headline)
    _patch_llm(monkeypatch, [_pick(quote=headline)])

    state = _state(seed)
    asyncio.run(nodes.quote(state))

    assert state["quotes"] == {}
    assert state["meta"]["quote"]["no_quote"] == 1


def test_quote_survives_llm_failure(monkeypatch, clean_db):
    seed = _seed()
    _patch_llm(monkeypatch, [TimeoutError("model hung")])

    state = _state(seed)
    asyncio.run(nodes.quote(state))

    assert state["quotes"] == {}
    assert state["meta"]["quote"]["failed"] == 1


def test_quote_falls_back_to_source_when_translation_loses_digits(monkeypatch, clean_db):
    article = (
        "The council debated the budget for an hour. "
        "The program cost 12 million dollars over two years and served four districts."
    )
    seed = _seed(article)
    _patch_llm(
        monkeypatch,
        [_pick(quote="The program cost 12 million dollars over two years.")],
        _translation(
            quote="El programa cost\u00f3 millones de d\u00f3lares durante dos a\u00f1os."
        ),
    )

    state = _state(seed)
    asyncio.run(nodes.quote(state))

    stored = state["quotes"][seed["item_id"]]
    assert stored["quote"] == "The program cost 12 million dollars over two years."
    assert stored["quote_translated"] is False
    assert state["meta"]["quote"]["translated"] == 0
    assert any("translation rejected" in w for w in state["warnings"])


def test_quote_skips_items_without_article_text(monkeypatch, clean_db):
    seed = _seed()
    with Session(engine) as s:
        item = s.get(NewsItem, uuid.UUID(seed["item_id"]))
        item.article_text = None
        s.add(item)
        s.commit()
    _patch_llm(monkeypatch, [_pick(quote=SOURCE_QUOTE)])

    state = _state(seed)
    asyncio.run(nodes.quote(state))

    assert state["quotes"] == {}
    assert state["meta"]["quote"]["skipped_no_text"] == 1
    assert state["meta"]["quote"]["attempted"] == 0


def test_quote_noop_without_llm(monkeypatch, clean_db):
    seed = _seed()
    monkeypatch.setattr(nodes, "chat_llm_configured", lambda: False)

    state = _state(seed)
    asyncio.run(nodes.quote(state))

    assert not state.get("quotes")
    assert state["meta"]["quote"]["status"] == "skipped"

