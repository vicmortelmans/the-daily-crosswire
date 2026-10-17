"""Quality + latency bench for multilingual drafting.

Runs the `draft` LLM call against qwen3:14b (or whatever is configured) in
each of the 6 supported languages on the SAME source article, then prints
the generated rendered_title + body with a small heuristic check that the
output is actually in the target language (via langdetect if available,
otherwise a coarse character/word check).

Run inside the backend container:

    docker compose exec backend python -m scripts.bench_multilang
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.agent.llm import _using_ollama, get_structured_llm
from app.agent.prompts import DRAFT_INSTRUCTION, SYSTEM_PROMPT
from app.agent.languages import SUPPORTED, language_name
from app.config import settings

TIMEOUT_SECONDS = 300


class _DraftOutput(BaseModel):
    rendered_title: str = Field(description="Replacement headline, under 70 chars.")
    body: str = Field(description="100-150 word body.")


# One Spanish-source article (fictional Gotham, rendered in Spanish as
# "Ciudad Gótica"). Translation from ES is the hardest case for each
# target language, so it's a good shared baseline.
SOURCE = {
    "topic_name": "Gotham",
    "title": "Ciudad Gótica reabre el puente del puerto tras seis meses de obras",
    "url": "https://example.com/gotham-bridge",
    "summary": (
        "Ciudad Gótica reabrió al tráfico el puente del puerto el lunes "
        "tras seis meses de obras de refuerzo estructural. La alcaldía "
        "informó que se reemplazaron veinte vigas de acero y se renovó "
        "el sistema de drenaje, lo que debería extender la vida útil del "
        "puente en treinta años. El corte había redirigido cerca de "
        "cuarenta mil vehículos diarios hacia rutas alternas, con demoras "
        "sustanciales en horas pico. Autoridades agradecieron la paciencia "
        "de los residentes y confirmaron que la próxima fase de obras "
        "urbanas se concentrará en la red de transporte público."
    ),
}


# Small heuristic: per language, a few function words we expect to appear
# frequently in a 100-150 word news body.
LANG_MARKERS = {
    "en": ("the", "of", "and", "that"),
    "es": ("el", "la", "los", "las", "que", "de"),
    "fr": ("le", "la", "les", "des", "que", "d'"),
    "it": ("il", "la", "dei", "delle", "che", "di"),
    "pt": ("o", "a", "os", "as", "que", "de"),
    "de": ("der", "die", "das", "und", "mit", "dass"),
    "nl": ("de", "het", "een", "en", "van", "op", "in", "met", "dat", "door"),
}


def _lang_hit_rate(text: str, code: str) -> float:
    markers = LANG_MARKERS.get(code, ())
    if not markers or not text:
        return 0.0
    lower = text.lower()
    tokens = re.findall(r"[a-zàâäæçéèêëîïôöœùûüÿñáéíóúüÁÉÍÓÚÑß]+", lower)
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in markers)
    return hits / len(tokens)


async def _draft_once(code: str) -> dict:
    name = language_name(code)
    user_msg = DRAFT_INSTRUCTION.format(
        topic_name=SOURCE["topic_name"],
        title=SOURCE["title"],
        url=SOURCE["url"],
        summary=SOURCE["summary"],
        language_name=name,
    )
    llm = get_structured_llm(_DraftOutput)
    t0 = time.perf_counter()
    try:
        result: _DraftOutput = await asyncio.wait_for(
            llm.ainvoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_msg)]),
            timeout=TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        elapsed = time.perf_counter() - t0
        return {"ok": False, "elapsed": elapsed, "error": "TIMEOUT"}
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {"ok": False, "elapsed": elapsed, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}

    elapsed = time.perf_counter() - t0
    body = (result.body or "").strip()
    title = (result.rendered_title or "").strip()
    wc = len(body.split())
    hit_rate = _lang_hit_rate(body, code)
    title_hit = _lang_hit_rate(title, code)
    return {
        "ok": True,
        "elapsed": elapsed,
        "title": title,
        "body": body,
        "word_count": wc,
        "lang_match_body": hit_rate,
        "lang_match_title": title_hit,
    }


async def main() -> None:
    mode = "ollama" if _using_ollama() else "openai-compat"
    model = settings.ollama_model if _using_ollama() else settings.openai_model
    print(f"mode={mode}  model={model}  timeout={TIMEOUT_SECONDS}s")
    print(f"source article (ES): {SOURCE['title']!r}")
    print("=" * 72)

    for code, name in SUPPORTED:
        print(f"\n[{code}] {name}")
        print("-" * 60)
        r = await _draft_once(code)
        if not r["ok"]:
            print(f"  {r['error']}  ({r['elapsed']:.1f}s)")
            continue
        wc_ok = 100 <= r["word_count"] <= 150
        print(f"  latency:   {r['elapsed']:.1f}s")
        print(f"  title:     {r['title']}")
        print(f"  word_count: {r['word_count']}  {'(OK)' if wc_ok else '(out of 100-150 bounds)'}")
        print(
            f"  lang markers. body: {r['lang_match_body']*100:.1f}%  title: {r['lang_match_title']*100:.1f}%"
        )
        print(f"  body:")
        for line in _wrap(r["body"], 68):
            print(f"    {line}")


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []
    n = 0
    for w in words:
        if n + len(w) + (1 if cur else 0) > width and cur:
            lines.append(" ".join(cur))
            cur, n = [w], len(w)
        else:
            cur.append(w)
            n += len(w) + (1 if len(cur) > 1 else 0)
    if cur:
        lines.append(" ".join(cur))
    return lines


if __name__ == "__main__":
    asyncio.run(main())
