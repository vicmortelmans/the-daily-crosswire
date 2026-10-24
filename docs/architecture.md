# Architecture

How the pieces fit, end-to-end. Read this after skimming the README.

## Services

Four containers, managed by a single `docker-compose.yml`:

1. **`db`**: Postgres 16 with the `pgvector` extension. Holds configs,
   topics, RSS feeds, news items (with 768-dimensional embeddings), issues,
   and a story log used for deduplication.
2. **`searxng`**: self-hosted SearXNG meta-search. JSON output enabled,
   rate limiter off. Serves as the LLM agent's web-search tool.
3. **`backend`**: FastAPI + LangGraph + APScheduler.
4. **`ui`**: Vite dev server + React + TanStack Query + Tailwind.
   Talks to the backend via `/api/*` (proxied through Vite).

The UI and backend also share `/issues/{slug}/`: FastAPI serves the
generated newspaper HTML as static files; the UI iframes it for preview.

## Data model

| Table                | What it holds                                                             |
| -------------------- | ------------------------------------------------------------------------- |
| `newspaper_configs`  | User-defined "newspapers": name, schedule, topics, active flag, optional max-stories-per-topic cap, `include_workflow_report` (write the optional per-article HTML report), plus `newsletter_emails` (list of recipient addresses for automatic delivery after each run). |
| `topics`             | Beats attached to a config: name, preferred domains, free-text search hints. |
| `rss_feeds`          | Per-topic RSS URLs, with etag + last-modified for cheap conditional polls. |
| `news_items`         | Ingested stories (URL-unique). Title, summary, `source_domain`, `topic_id`, `embedding vector(768)`. Also captures how the item was discovered: `ingested_via` (`'rss'` or `'searxng'`), `source_feed_url` (when from a feed), `source_query` (when from web search). Legacy rows pre-source-tracking have these as null. |
| `issues`             | One generated newspaper: status, slug, issue_date, lookback window, html_path, `drafts_json` (full pipeline state). |
| `story_log`          | One row per drafted story per issue. Holds the rendered title, rendered body, and `rendered_title_embedding`. Semantic dedup queries against this on the next run. |

Foreign keys:

- `news_items.topic_id` and `story_log.topic_id` are `ON DELETE SET NULL`
  so deleting a config cascades cleanly without blocking on referenced
  rows.
- `issues -> newspaper_configs` and `story_log -> issues` cascade-delete at
  the ORM level (`cascade: all, delete-orphan` in SQLModel).

pgvector indexes: `ivfflat` with `vector_cosine_ops` on both vector
columns. Uses cosine distance (`<=>`) for similarity queries.

## The agent pipeline (14 nodes)

When a config fires (on schedule or from the "Generate now" button) the
endpoint creates a pending `issues` row, captures its id, and enqueues a
one-shot APScheduler job pointed at that issue id. The job calls the
LangGraph graph. Actual execution order after `dedupe` is
`fetch_articles -> select -> draft`; the descriptions below keep the
selection and fetch sections together for readability. Nodes in order:

1. **`ingest_rss`**: for every topic with RSS feeds, `feedparser.parse`
   the URL (with etag/last-modified for conditional requests). New entries
   are upserted into `news_items`. The feed row's `last_polled_at`,
   `last_etag`, `last_modified` are updated.

2. **`ingest_search`**: for topics with no feeds or too-few-fresh-items
   (`< 3` in the lookback window), call SearXNG with the topic name plus
   optional `site:` filters from `preferred_sources` and free-text
   `search_hints`. Results are bulk-inserted with `ON CONFLICT DO NOTHING`
   on `url` so concurrent runs cannot race.

3. **`embed`**: batch-embed any `news_items` in the lookback window that
   don't have an embedding yet. Default model is `nomic-embed-text`
   (768 dim) via local Ollama. Records `{total_candidates, newly_embedded,
   unembedded}` in `state.meta.embedding` so a silent embedder outage
   doesn't quietly downgrade dedup to URL-only.

4. **`dedupe`**: filter candidates against `story_log` over a window of
   `max(window_days * 4, 14)` days. Rules fire in order; first match wins:
   1. Exact URL match against already-drafted stories.
   2. Exact `normalized_title` match (lowercase, punctuation stripped).
   3. Cosine similarity > `DEDUP_SIM_THRESHOLD` (default 0.88) against
      `story_log.rendered_title_embedding`. Uses pgvector's `<=>` operator.

5. **`select`**: LLM quality filter with **batching, per-batch cap, and a
   two-pass intra-topic dedup** on the output.

   For each topic, the dedup-survivors are split into chunks of
   `SELECT_BATCH_SIZE` (default 10). Each chunk goes to a separate LLM
   call that returns a flat `_SelectOutput { keep_ids, lead_id }`. The
   prompt has two rubrics (Entertainment vs Local/City News) and asks the
   LLM to rank best-first and return **at most `SELECT_MAX_KEPT_PER_BATCH`
   items** (default 5). The code also slices to the same cap as a safety
   net, respecting the LLM's ordering so if the LLM obeys the prompt
   nothing is lost.

   `temperature=0` on this node; filtering is a decision task, not creative
   writing. Each LLM call is wrapped in a retry loop (`_invoke_with_parse_retry`,
   up to `LLM_MAX_RETRIES=3` attempts, `LLM_TIMEOUT_SECONDS=300s` each).
   On exhaustion the batch is **dropped** (not passed through): a batch
   that can't produce valid JSON three times in a row is more likely to
   leak junk than to contain gold.

   After batches merge, two intra-topic dedup passes run per topic:
   1. **LLM dedup pass** (`DEDUP_INSTRUCTION`): sees all surviving items
      with title + summary together, clusters by "same event," and returns
      `drop_ids`. Fixes the "same story covered by three outlets" pattern
      that batching can't see (each batch only sees a slice). When picking
      a cluster's representative, the prompt prefers the most specific
      item (drops aggregator/landing-page URLs in favor of the concrete
      story).
   2. **Embedding safety net** (`_collapse_similar_embeddings`): re-embeds
      `title + summary[:500]` on the fly (not the stored title-only
      vector), greedy-collapses survivors above
      `SELECT_INTRA_SIM_THRESHOLD` (default 0.75). Cheap, catches anything
      the LLM missed.

   Finally, if the config has `max_stories_per_topic` set, the surviving
   list is clipped; otherwise the full list passes through.

6. **`fetch_articles`**: retrieves full article text for every candidate
   that survived cross-issue dedup, before the quality filter runs. Uses
   `trafilatura` (with fallback heuristics)
   over async HTTP with realistic browser headers and bounded concurrency.
   Extracted text is cached in `news_items.article_text` so repeat runs skip
   external requests, and is hard-capped at `ARTICLE_MAX_CHARACTERS`
   (default 3000) to keep select/draft prompts within token and timeout
   budgets. If a site is blocked, paywalled, or times out, the node records
   status `fallback_summary` and gracefully falls back to the RSS/search
   snippet.

7. **`draft`**: per selected story, the LLM writes a replacement headline
   (< 70 chars) and a 100-150 word body in cause-and-effect form in the
   config's configured **output language** (`en`, `es`, `fr`, `it`, `pt`,
   `de`). It operates on the full article text (or fallback summary),
   translating from multilingual sources as needed. Voice rules
   (no hype, no exclamation, no emoji) are enforced by the system prompt.

8. **`critique`**: Python pre-check scans drafts for voice violations:
   - Exclamation marks and word-count bounds are **language-agnostic**.
   - Hype-word detection uses a per-language list (see
     [backend/app/agent/languages.py](../backend/app/agent/languages.py)).
     Examples: `revolucionario` for Spanish, `révolutionnaire` for French,
     `bahnbrechend` for German, so the mechanical gate works in every
     supported language.

   Flagged drafts are sent to the LLM for rewrite; clean drafts skip the
   round-trip. Rewrites stay in the same language as the draft.

9. **`editorial`**: one 200-word editorial drawing from the lead headlines
   across topics, written in the config's output language. Optional;
   skipped if no leads.

10. **`puzzles`**: themed puzzles page. Pool of words derived from the
   drafted article bodies and topic names; passed to
   [backend/app/puzzles/build.py](../backend/app/puzzles/build.py). Six
   puzzles (sudoku, criss-cross fill-in, word search, maze, 4x4 tic-tac-toe,
   cryptogram), each verified solvable before HTML is emitted. Deterministic
   seed derived from `sha256(issue_id | config_id)[:4]` and persisted in
   `drafts_json.puzzles_meta.seed` so re-renders are byte-identical.

11. **`finalize`**: persists the whole pipeline state
    (`drafts_json = {drafts, editorial, warnings, meta, puzzles_meta}`) on
    the `issues` row and sets `status = drafted` (or `failed` if no drafts
    were produced).

12. **`render`**: renders [issue.html.j2](../backend/app/templates/issue.html.j2)
    into `issues/{slug}/index.html`, copies the shared `style.css`, rebuilds
    `story_log` rows with rendered-title embeddings, and marks `status = ready`.

13. **`notify`**: if `config.newsletter_emails` is non-empty and the issue
    is `ready`, renders the HTML to PDF via Playwright + headless Chromium
    ([backend/app/services/pdf.py](../backend/app/services/pdf.py)) and
    mails it to every recipient as a multipart/alternative message with
    attachment ([backend/app/services/mail.py](../backend/app/services/mail.py)).
    Skips cleanly when the recipient list is empty or SMTP is unconfigured.
    A failed send does not flip issue status; the newspaper is already
    published. Per-recipient outcomes and the overall status land in
    `meta.notify`.

14. **`print`**: optional CUPS printer submission if configured. Like
    `notify`, a print failure does not flip the issue status.

On an Apple M4 Pro with 48 GB unified RAM running qwen3:14b locally, the
whole run takes 15-30 minutes depending on LLM latency and how many
stories survived dedup. Hosted providers cut that by a large factor.
The `notify` node adds a few seconds (Chromium launch + PDF render + SMTP
round trip), independent of LLM latency.

## The lookback contract

Which news counts as "today's" for a given run is decided in one place:

- `runner.py::_compute_lookback` decides the window when the Issue row is
  created, before the agent runs:
  - **First run of a config:** interval mode falls back to
    `today - frequency_days`; weekdays mode to `today - 7 days`.
  - **Subsequent runs:** start where the last completed issue left off.
    This handles skipped runs cleanly: the next run covers the gap, not a
    fixed 3-day or 7-day slice.
- The window is persisted on the `issues` row as `lookback_start` and
  `lookback_end`.
- `_load_issue_window(session, state)` in
  [nodes.py](../backend/app/agent/nodes.py) reads those fields back. Every
  node that asks "which items count for this run?" uses the same answer:
  `ingest_search`, `embed`, and `dedupe` all go through this helper.

No other code reads `config.frequency_days` to derive a window.

## Scheduler

[backend/app/scheduler.py](../backend/app/scheduler.py) uses a single
`AsyncIOScheduler` with a `SQLAlchemyJobStore` pointing at the same
Postgres. Jobs survive backend restarts. Three kinds of jobs:

1. **Global RSS poll** (`rss-poll-global`): runs every
   `RSS_POLL_INTERVAL_HOURS` hours (default 1) to keep the news_items pool
   warm across all configs.
2. **Per-config generation** (`config-{id}`): registered on config
   save/update, removed on delete or when `active = false`:
   - Interval mode: `IntervalTrigger(days=N, start_date=<next UTC run_hour>)`.
     Anchoring to the next occurrence of `run_hour` means every fire lands
     on the same UTC hour rather than drifting by whatever time the job
     happened to be registered.
   - Weekdays mode: `CronTrigger(day_of_week=..., hour=run_hour)`.
3. **Manual runs** (`issue-{id}-manual-{ts}`): the "Generate now" endpoint
   creates a pending Issue synchronously, returns its id, and fires a
   one-shot `DateTrigger` against that specific issue.

On every backend startup, all `active` configs are re-registered from the
DB. The source of truth is `newspaper_configs.active`, not whatever the
jobstore happens to hold.

### Agent worker loop

[backend/app/agent/worker.py](../backend/app/agent/worker.py) owns a single
long-lived asyncio event loop on a dedicated daemon thread. Scheduler jobs
(`run_generation_job`, `run_issue_job`), the debug `/api/debug/agent` endpoint,
and `/api/debug/embed` all submit their coroutine to that loop through
`agent_worker.run()` / `agent_worker.run_blocking()`.

This matters because langchain/OpenAI clients and the httpx connection pools
they hold are bound to the event loop that created them. Running each job with
`asyncio.run()` gave every run a fresh loop and closed it afterwards; the next
run then reused cached keep-alive connections from the dead loop and failed
with `RuntimeError: Event loop is closed`. The worker loop starts in the app
lifespan before the scheduler and is stopped after the scheduler shuts down.
Runs are serialized on the worker loop (semaphore of 1) so two configs cannot
interleave and share client state.

A full run is also bounded by `RUN_TIMEOUT_SECONDS` (default 7200) around
`graph.ainvoke`; expiry marks the issue `failed` with an explicit timeout
error. Scheduler job wrappers raise `AgentRunFailed` when the issue ends up
`failed`, so APScheduler records a failed job instead of logging
"executed successfully".

## Renderer

[backend/app/services/renderer.py](../backend/app/services/renderer.py)
reads `issues.drafts_json`, groups drafts by topic, picks a hero (lead of
the topic with the most stories), and renders
[backend/app/templates/issue.html.j2](../backend/app/templates/issue.html.j2).

Layout (uses the legacy newspaper's class names: `.nameplate`, `.deck`,
`.hed`, `.kicker`, `.byline`, `.dropcap`, `.broadsheet`, `.col`,
`.section-rule`, `.editorial`, so the print stylesheet works as-is):

- Masthead: `.edition-line`, `.nameplate`, `.date-bar`.
- Hero block (`.deck` with `.hed-hero`): the top topic's lead, shown as a
  single-sentence dek rather than the full body (body renders in the
  beat section below).
- The hero topic's beat is reordered to render **first**, so the front-page
  lead and its related stories live on the same page.
- Per-topic `.broadsheet`: `.col.lead-story` (with `.dropcap` on the
  first paragraph) plus a `.col-group` of `.col` articles.
- `<hr class="section-rule">` between topics.
- Editorial block (`.editorial` with `.kicker.kicker-red`).
- Small colophon with `Issue seed / Generated / Lookback` metadata (no
  source list; each story already links its own source inline).
- Trailing `<!-- GAMES_PAGE_START -->` and `<!-- GAMES_PAGE_END -->` markers
  wrapping the puzzles fragment.

The whole document is wrapped in `<main class="paper">`. The stylesheet
makes `<body>` a dark "desk" background and `.paper` the light sheet
floating on top. Without the wrapper the content disappears into the desk
colour.

Page count is not fixed. The layout flows naturally across however many
US Letter pages the content needs when printed. The browser handles pagination
via CSS multi-column (`columns: 3` on `.col-group`) and `break-before: page`
on the games section.

The style sheet is copied from the repo root (`style.css`) into
`issues/{slug}/style.css` on every render, so changes to the shared
template apply to newly-rendered issues. Existing issues keep their
snapshot; re-render to pick up style changes.

**Localization.** The renderer reads `config.language` and pulls template
strings and long-form date formatting from
[backend/app/agent/languages.py](../backend/app/agent/languages.py):
masthead edition line, "By Desk Wire", "From the editors", "Source:",
"Issue seed / Generated / Lookback", "Print", "Vol. / No.", weekday and
month names. Supported: `en`, `es`, `fr`, `it`, `pt`, `de`. Adding a new
language means: add it to `SUPPORTED`, fill in `_TEMPLATE_STRINGS`,
`_MONTHS`, `_WEEKDAYS`, optionally `_HYPE_WORDS`, and the `LanguageCode`
Literal in [backend/app/schemas.py](../backend/app/schemas.py).

### PDF generation (newsletter path only)

Interactive users Cmd+P the iframe preview to save a PDF. For the
newsletter attachment the `notify` node calls
[backend/app/services/pdf.py](../backend/app/services/pdf.py), which
launches headless Chromium via Playwright, loads the rendered
`issues/{slug}/index.html` via `file://` URL, and calls `page.pdf()` with
`print_background=True` + `prefer_css_page_size=True` and zero margins.
The existing `@page` rules in `backend/app/templates/style.css` drive
pagination exactly as they do in Chrome under Cmd+P: named pages (news vs
games), letter size, columns, and Google Fonts all carry over.

Playwright is installed in the backend Dockerfile
(`playwright install chromium --with-deps`); this adds ~400 MB to the
image. Acceptable for a single-user local stack. CLAUDE.md scopes the
rule to this one file so no other pipeline stage introduces a headless
browser.

## Puzzle generation

[backend/app/puzzles/build.py](../backend/app/puzzles/build.py) is the
single source for all six generators. Public signature:

```python
build_puzzles(
    seed: int,
    theme_pool: dict | list | None = None,   # defaults used if None
    wordsearch_pool: list[str] | None = None,
    crypto_phrases: list[str] | None = None,
    target_givens: int = 54,
    with_fallback: bool = True,               # revert to default pools on failure
) -> tuple[str, dict]
```

Every generator has a verifier. The sudoku backtracker confirms exactly
one solution; the criss-cross builder brute-forces word-to-slot permutations
to confirm uniqueness; the word-search scanner re-finds every placed word
in all eight directions; BFS confirms the maze is solvable; the cryptogram
roundtrip-decodes and asserts every hint letter appears in the encoded
text. Every verifier is an `assert`; if any fails, the function raises.

`with_fallback=True` (the default, used by the agent) means a dynamic
pool that can't produce a solvable puzzle falls back to the hardcoded
`DEFAULT_*` pools. `with_fallback=False` (used by the legacy CLI) crashes
loudly instead. The paper always ships either way.

## Dedup memory

`story_log` is the long-term memory. For each story that lands in a
published newspaper, a row records:

- Which issue it appeared in.
- The source `news_item.id` (and by extension the URL).
- The title as it appeared in the paper (after LLM rewrite).
- An embedding of that rendered title.

On the next run, the dedup node queries `story_log` joined with `news_items`
over the lookback window and tests candidates against it. The pgvector
index makes similarity search sub-linear.

## Metadata captured per run

Every node writes structured telemetry into `issues.drafts_json.meta`.
The [issue details dashboard](../ui/src/pages/IssueDetails.jsx) reads
this via `GET /api/issues/{id}/details` and renders it across six tabs
(Overview, Pipeline, Drops, Timings, Logs, Run info).

**Mid-run checkpointing.** The `_timed` wrapper in
[backend/app/agent/graph.py](../backend/app/agent/graph.py) flushes the
entire `meta` dict to `issues.drafts_json` at the **entry** and **exit**
of every node, not just once at the end. The dashboard polls the
details endpoint every 2 s while `status = running`, so the user sees
the pipeline fill in live: which node is currently running, elapsed
time on that step, completed nodes, drop counts so far, warnings as they
fire. On failure, the wrapper's last write captures `failed_at_node` and
`error` so the failed-issue banner has something to show. Older issues
generated before this landed still render (with "Not available" for
anything missing); no migration was needed.

Keys under `meta`:

| Key              | Contents |
| ---------------- | -------- |
| `run`            | Chat model and endpoint, embedding model, tunables snapshot (`select_batch_size`, `select_max_kept_per_batch`, `select_intra_sim_threshold`, `dedup_sim_threshold`, `llm_timeout_seconds`, `llm_max_retries`), `started_at`, `finished_at` |
| `timings`        | Per-node wall-clock duration in milliseconds (ingest_rss, ingest_search, embed, dedupe, select, draft, critique, editorial, puzzles). `finalize` and `render` are not captured because those nodes persist to DB before the timing wrapper records duration. |
| `llm_calls`      | Per-stage cumulative stats: `{count, total_duration_ms, retries, timeouts, parse_errors, exhausted}`. Stages: `select`, `intra_dedup`, `draft`, `critique`, `editorial`. |
| `ingest_rss`     | `{feeds_polled, items_new, failures}` |
| `ingest_search`  | `{topics_queried, items_added, failures}` |
| `embedding`      | `{total_candidates, newly_embedded, unembedded}` |
| `dedup_breakdown`| `{total, kept, dropped_url, dropped_title, dropped_similarity}` |
| `select`         | `{total_kept, topics_with_keeps, topics: {topic_id: {topic_name, candidates, batches, kept, collapsed_total, has_lead}}}` |
| `draft`          | `{attempted, succeeded, failures}` |
| `critique`       | `{checked, flagged, rewrites, failures, llm}` |
| `editorial`      | `{status: ok/failed/skipped, word_count, error?}` |
| `puzzles`        | `{status: ok/fallback/failed/skipped, seed, theme_pool_size, crypto_phrase_candidates}` |
| `notify`         | `{status: ok/partial/skipped/misconfigured/failed, sent?, failed?, per_recipient?: [{email, status, error?}], reason?, error?}`. Records the automatic newsletter delivery for this run. `skipped` when no subscribers or issue not ready; `misconfigured` when SMTP is unset; `partial` when some recipients failed. |
| `funnel`         | Flat pipeline counts: `ingested_new -> embedded_candidates -> after_cross_dedup -> after_select_and_intra_dedup -> drafted -> editorial_ok / puzzles_status` |
| `drop_trace`     | List of every dropped candidate. See below. |
| `current_node`   | Name of the node currently in flight, or null after notify completes. Set on node entry, cleared on notify exit. |
| `completed_nodes`| Append-only list of node names that have finished. Drives the green checkmarks on the Pipeline Progress panel. |
| `node_started_at`| Map of node name to ISO timestamp of when that node started. Used by the dashboard to show live elapsed-on-this-step while a run is active. |
| `failed_at_node` | On failure, the node that was in flight when the graph raised. |
| `error`          | Short error string mirror of `issues.error`, kept inside `meta` so the drop-trace and dashboard show it in context. |

### `drop_trace` entries

Each drop entry:

```json
{
  "news_item_id": "<uuid>",
  "stage": "select_llm",
  "topic_id": "<uuid>",
  "reason": "trailer-only, no near-term release",
  "snapshot": {
    "title": "...",
    "url": "https://...",
    "source_domain": "showwire.example",
    "summary": "first 300 chars of the original summary",
    "published_at": "2026-04-22T14:00:00+00:00",
    "ingested_via": "rss",
    "source_feed_url": "https://example.com/feed.xml",
    "source_query": null
  }
}
```

`snapshot` carries a denormalized copy of the `news_items` row so drops
stay legible even if the source row is garbage-collected later. Legacy
drops (written before source tracking landed) don't have the last three
fields; the UI omits the "via ..." line for those.

Stages and their reason sources:

| Stage                   | When fires                                                                    | Reason written by |
| ----------------------- | ----------------------------------------------------------------------------- | ----------------- |
| `dropped_url`           | cross-issue dedup, URL already in `story_log` window                          | synthesized        |
| `dropped_title`         | cross-issue dedup, normalized title already in `story_log` window             | synthesized        |
| `dropped_similarity`    | cross-issue dedup, cosine > `DEDUP_SIM_THRESHOLD` vs recent `story_log`       | synthesized with score + matched title |
| `select_llm`            | quality filter LLM chose not to keep                                          | **LLM's `drop_reasons`** |
| `select_cap`            | LLM kept but trimmed to `SELECT_MAX_KEPT_PER_BATCH`                           | synthesized        |
| `intra_dedup_llm`       | intra-issue dedup pass 1 (LLM clustering)                                     | **LLM's `drop_reasons`** |
| `intra_dedup_emb`       | intra-issue dedup pass 2 (embedding safety net)                               | synthesized with sim score + kept id |
| `select_topic_cap`      | `config.max_stories_per_topic` trimmed it                                     | synthesized        |

The inline `snapshot` means a drop entry stays readable in the dashboard
even after a cleanup job deletes the `news_items` row it references.

### Workflow report (optional)

When `config.include_workflow_report` is true, `runner.run_issue()` seeds a
separate `AgentState["workflow_report"]` dictionary. Nodes append events in
workflow order; the trace is intentionally *not* written to `drafts_json` or
any database table.

Implementation:
[backend/app/services/workflow_report.py](../backend/app/services/workflow_report.py)
and [backend/app/templates/workflow_report.html.j2](../backend/app/templates/workflow_report.html.j2).

Each event carries `news_item_id`, `step`, `status`, an optional reason (with
`reason_source` = `ai` or `system`), and a step-specific `data` payload
(title/summary for retrieval, extracted article text, draft body, critique
before/after, similarity scores, batch ranks, and so on).

Writers are called at node boundaries (`graph._timed`) and at sub-step
progress checkpoints (`nodes._checkpoint_progress`), throttled to one write
every few seconds. Each write atomically replaces:

- `issues/{slug}/workflow-report.html` — article-oriented standalone report
- `issues/{slug}/workflow-report.json` — workflow-order event ledger
- `issues/{slug}/workflow-report.meta.json` — tiny status file read by the
  details endpoint

The dashboard's **Report** tab reads availability from
`GET /api/issues/{id}/details`, iframes the HTML, and offers a download link.
The report is marked **partial** while the run is active and **final** after
the last node. No report artifacts are written for configs with the flag off.

### Per-run log buffer

Separately from `meta`, each active run gets an in-memory log buffer
scoped to its `issue_id`. Implementation in
[backend/app/services/log_buffer.py](../backend/app/services/log_buffer.py):

- A `ContextVar` set by `runner.run_issue()` before `graph.ainvoke` and
  cleared in a `finally`. Every log record emitted while that task is on
  the stack gets routed to the right issue's buffer.
- A filter drops `uvicorn.access` entirely (all the `/api/issues/*`
  polling the UI itself generates), `httpx` INFO (the `200 OK` spam from
  every LLM + embedding call), and any message matching
  `/api/(health|issues|configs|debug/scheduler)`. Kept: any record from
  `app.*` at INFO+, anything from anywhere at WARNING+.
- Buffers cap at 500 entries per issue and TTL 10 minutes after the
  last write. Finished runs stay viewable briefly then evict
  themselves on the next request.
- Served by `GET /api/issues/{id}/logs?since=<iso>`. The dashboard's
  Logs tab polls this every 2 s while the run is active and appends
  new entries in place.

Because this is in-memory and single-process, a backend restart wipes
the buffer. That's acceptable: a restart kills any in-flight
generation anyway, and historical logs live in the backend container's
stdout (`docker compose logs backend`).

## LLM transport

Two code paths, chosen by env vars in `app/agent/llm.py::_using_ollama()`:

- **Default: `langchain-openai` (`ChatOpenAI`)** talking to any
  OpenAI-compatible endpoint: hosted (OpenAI, OpenRouter, Groq) or local
  (vLLM, LM Studio, Ollama via its OpenAI-compat shim).
- **Ollama native override: `langchain-ollama` (`ChatOllama`)**, activated
  when BOTH `OLLAMA_API_BASE` (no `/v1`) and `OLLAMA_MODEL` are set.
  Structured-output calls go through Ollama's real `/api/chat` tool-calling
  path rather than the OpenAI-compat shim, which has been unreliable with
  some gateway proxies (see
  [docs/reports/litellm-function-calling-hang.md](reports/litellm-function-calling-hang.md)).
  Use this whenever the target is a local Ollama instance; it is materially
  more reliable for structured output.

`_invoke_with_parse_retry` wraps every structured call with
`asyncio.wait_for(timeout=LLM_TIMEOUT_SECONDS)` and up to
`LLM_MAX_RETRIES` attempts on `asyncio.TimeoutError` or
`OutputParserException`. This is runtime-level timeout enforcement.
Some endpoints (Ollama native, certain proxies) can silently hang with
open TCP and no data flowing, so SDK-level timeouts alone don't always
save you.

## Voice

The system prompt lives in
[backend/app/agent/prompts.py](../backend/app/agent/prompts.py). Key
rules (see [voice.md](voice.md) for the full account):

- No hype, no superlatives, no exclamation marks, no emoji.
- Cause and effect: "X happened because Y, which means Z."
- No fabricated quotes, stats, or details beyond what the source supports.
- Draft body: 100-150 words. Editorial: up to 200 words.
- PR dressed as news, rumor-only posts, and rehashes are dropped.

Critique pass catches violations mechanically before involving the LLM
again. See [docs/voice.md](voice.md) for the full rules.
