import asyncio
import logging
import re
from typing import Optional

import httpx

try:
    import trafilatura

    _HAS_TRAFILATURA = True
except ImportError:
    _HAS_TRAFILATURA = False

try:
    from bs4 import BeautifulSoup

    _HAS_BS4 = True
except ImportError:
    _HAS_BS4 = False

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,es;q=0.8,fr;q=0.7,de;q=0.6,it;q=0.5,nl;q=0.4,*;q=0.3",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "cross-site",
}

_WS_RE = re.compile(r"\n{3,}")
MIN_WORD_COUNT = 40


def extract_text_from_html(html: str) -> Optional[str]:
    """Extract clean article body text from raw HTML.

    Prefers trafilatura when available; falls back to BeautifulSoup-based
    heuristic extraction. Returns None if extracted text is below the
    minimum word threshold (indicating an empty shell, cookie wall, or
    paywall gate).
    """
    if not html or not html.strip():
        return None

    text: Optional[str] = None

    if _HAS_TRAFILATURA:
        try:
            text = trafilatura.extract(
                html,
                output_format="txt",
                include_comments=False,
                include_tables=False,
                no_fallback=False,
            )
        except Exception as exc:
            logger.debug("trafilatura extraction failed: %s", exc)
            text = None

    if (not text or len(text.split()) < MIN_WORD_COUNT) and _HAS_BS4:
        try:
            soup = BeautifulSoup(html, "lxml" if "lxml" in BeautifulSoup.__dict__ else "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript", "svg", "form"]):
                tag.decompose()

            # Target common article containers first
            container = soup.find("article") or soup.find("main") or soup.find("div", class_=re.compile(r"article[-_]body|story[-_]content|post[-_]body", re.I))
            root = container if container else soup.body or soup

            paragraphs = [p.get_text(separator=" ", strip=True) for p in root.find_all(["p", "h2", "h3", "h4", "blockquote"])]
            fallback_text = "\n\n".join(p for p in paragraphs if len(p.split()) >= 4)
            if len(fallback_text.split()) >= MIN_WORD_COUNT:
                text = fallback_text
        except Exception as exc:
            logger.debug("bs4 fallback extraction failed: %s", exc)

    if not text:
        return None

    cleaned = _WS_RE.sub("\n\n", text.strip())
    if len(cleaned.split()) < MIN_WORD_COUNT:
        return None

    return cleaned


async def fetch_article_text(
    url: str,
    *,
    client: Optional[httpx.AsyncClient] = None,
    timeout: float = 10.0,
) -> Optional[str]:
    """Fetch URL and extract the main article body.

    Returns the extracted text on success, or None on any network error,
    HTTP status error, or parsing failure.
    """
    if not url or not (url.startswith("http://") or url.startswith("https://")):
        return None

    async def _do_fetch(c: httpx.AsyncClient) -> Optional[str]:
        try:
            resp = await c.get(url, timeout=timeout, follow_redirects=True)
            if resp.status_code != 200:
                logger.warning("article fetch for %s returned HTTP %d", url, resp.status_code)
                return None

            content_type = resp.headers.get("content-type", "").lower()
            if content_type and "text/html" not in content_type and "application/xhtml" not in content_type:
                logger.debug("article fetch for %s has non-HTML content-type: %s", url, content_type)
                # Some servers omit or mislabel text/html, but if it's explicitly e.g. application/pdf, skip
                if any(x in content_type for x in ("application/pdf", "image/", "audio/", "video/")):
                    return None

            html = resp.text
            return await asyncio.to_thread(extract_text_from_html, html)
        except httpx.HTTPError as exc:
            logger.warning("article fetch HTTP error for %s: %s", url, exc)
            return None
        except Exception as exc:
            logger.warning("article fetch error for %s: %s", url, exc)
            return None

    if client is not None:
        return await _do_fetch(client)

    async with httpx.AsyncClient(headers=DEFAULT_HEADERS) as own_client:
        return await _do_fetch(own_client)


async def batch_fetch_articles(
    urls: list[str],
    *,
    max_concurrency: int = 5,
    timeout: float = 10.0,
) -> dict[str, Optional[str]]:
    """Fetch multiple article URLs concurrently with bounded concurrency.

    Returns a dict mapping url -> extracted text (or None if failed).
    """
    sem = asyncio.Semaphore(max_concurrency)
    results: dict[str, Optional[str]] = {}

    async with httpx.AsyncClient(headers=DEFAULT_HEADERS) as client:

        async def _fetch_one(u: str):
            async with sem:
                text = await fetch_article_text(u, client=client, timeout=timeout)
                results[u] = text

        tasks = [_fetch_one(u) for u in set(urls) if u]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    return results

