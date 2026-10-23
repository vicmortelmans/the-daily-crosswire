# Operations

Running tests, CLIs, debug endpoints, data maintenance,
and a troubleshooting catalog. For first-time setup see
[setup.md](setup.md).

## Running tests

The test suite is real-stack: it hits a live backend container, which
hits real RSS, real SearXNG, real embeddings (Ollama or whatever
`EMBEDDING_API_BASE` points at), and a real chat LLM. No mocks.

```bash
docker compose exec -T backend pytest -v
```

Typical run on an Apple M4 Pro with 48 GB unified RAM: 20-40 minutes.
LLM latency dominates.

**What's covered:**

- `tests/test_e2e_agent.py`: full pipeline runs (ingest, dedup, select,
  draft, critique, puzzles, render). Three tests cover the RSS path,
  the SearXNG-only path (feedless topic), and dedup against a seeded
  `story_log`.
- `tests/test_renderer.py`: renderer unit tests with fabricated
  `drafts_json`. Fast (no LLM). Tests atomicity and idempotent re-render.
- `tests/test_workflow_report.py`: pure unit tests for event grouping,
  outcome resolution, HTML escaping, artifact writing, and the disabled flag.
  No database or LLM calls.
- `tests/test_puzzles.py`: all six generators, determinism, fallback
  behavior.
- `tests/test_scheduler.py`: config CRUD and scheduler job lifecycle. Also
  a real end-to-end "scheduled job actually fires and completes" test
  (~20 seconds when the LLM drops everything, longer when it doesn't).

**Test hygiene.** The `clean_db` fixture only wipes rows whose config name
starts with `__test`. It also removes the corresponding `issues/{slug}/`
directories. Your own configs are never touched. Every test MUST use the
`__test_` prefix for its config names, otherwise cleanup will not fire.

```python
# good
CFG_NAME = "__test_e2e_agent"

# bad; leaks into your dev DB
CFG_NAME = "smoke-test"
```

## CLIs

### Alembic (migrations)

```bash
# Status:
docker compose exec backend alembic current

# Apply pending:
docker compose exec backend alembic upgrade head

# Roll back one:
docker compose exec backend alembic downgrade -1

# Autogenerate a new revision (always review before applying):
docker compose exec backend alembic revision --autogenerate -m "describe change"
```

**Caveat:** autogenerate tries to drop `apscheduler_jobs` and the ivfflat
vector indexes on every run because they're not declared in the SQLModel
metadata. Strip those from the generated file before applying. See
`backend/alembic/versions/75af7029299d_...` for the shape of a clean
hand-edited migration.

### Psql

```bash
docker exec -it newspaper-db-1 psql -U np -d newspaper
```

Helpful queries:

```sql
-- Configs and their schedules
SELECT name, schedule_kind, frequency_days, weekdays, run_hour, active
FROM newspaper_configs;

-- Recent issues with status
SELECT slug, status, issue_date,
       jsonb_array_length(drafts_json->'drafts') AS drafts
FROM issues
ORDER BY created_at DESC LIMIT 10;

-- Dedup lookback: what's in story_log for the last 14 days
SELECT sl.rendered_title, i.issue_date, i.slug
FROM story_log sl
JOIN issues i ON i.id = sl.issue_id
WHERE i.issue_date >= current_date - interval '14 days'
ORDER BY i.issue_date DESC;

-- Feed freshness
SELECT t.name AS topic, f.url, f.last_polled_at
FROM rss_feeds f JOIN topics t ON t.id = f.topic_id
ORDER BY f.last_polled_at DESC NULLS LAST;

-- How many news_items per topic
SELECT t.name, count(n.id) AS items
FROM topics t LEFT JOIN news_items n ON n.topic_id = t.id
GROUP BY t.name ORDER BY items DESC;
```

## Benchmark scripts

Three scripts under `backend/scripts/` exercise specific parts of the
pipeline against the configured LLM. They run inside the backend container
so they pick up the same `.env` and imports as the agent.

| Script                       | What it measures                                                  |
| ---------------------------- | ----------------------------------------------------------------- |
| `bench_llm.py`               | Raw latency of three call shapes (plain chat / JSON mode / structured native-tools). Useful to confirm the LLM transport is healthy before iterating on prompts. |
| `bench_select_quality.py`    | Labeled precision / recall of the `SELECT_INSTRUCTION` prompt on two hand-curated batches (entertainment + local-city). Prints false positives and false negatives by title. |
| `bench_select_reasons.py`    | Compares the baseline `_SelectOutput` schema against the extended `{keep_ids, lead_id, drop_reasons}` schema. Confirms the model produces valid drop reasons per item without losing select-decision stability or regressing latency significantly. |
| `bench_dedup.py`             | Labeled precision / recall of the two-pass intra-topic dedup (LLM pass + embedding safety net). Checks that every dup cluster collapses to exactly one survivor and that aggregator / landing pages don't win over concrete stories. |
| `bench_multilang.py`         | Runs the draft LLM call on the same Spanish source article in each of the 6 supported output languages (`en`, `es`, `fr`, `it`, `pt`, `de`). Reports latency, word count, a simple language-marker hit rate on the generated body, and the generated title. Fast sanity check that qwen3:14b (or whatever is configured) translates cleanly. |

```bash
docker compose exec backend python -m scripts.bench_llm
docker compose exec backend python -m scripts.bench_select_quality
docker compose exec backend python -m scripts.bench_select_reasons
docker compose exec backend python -m scripts.bench_dedup
docker compose exec backend python -m scripts.bench_multilang
```

Each script takes a few minutes end-to-end depending on LLM latency. Run
after any change to `prompts.py` or `nodes.py` to catch regressions before
triggering a full generation.

## Issue details dashboard

Every issue exposes a structured metadata dump at
`GET /api/issues/{id}/details`. The UI renders it as a six-tab dashboard
(`Overview`, `Pipeline`, `Drops`, `Timings`, `Logs`, `Run info`) at
`http://localhost:5173/issues/{id}/details`, linked from the newspapers
list via the **Details** button on each row. The regular **Open** button
on non-ready issues (pending, running, failed) redirects here too, since
there is no point showing an empty preview.

What's there:
- **Overview**:
  - While a run is active, a "Generation in progress" banner with the
    current node, per-step elapsed, and completed/total step counter.
  - A **Pipeline Progress** grid for all 13 dashboard nodes: green check for
    completed (with the stored duration), blue spinner for in-flight
    (with live elapsed ticking), grey for pending.
  - The count **funnel** and the per-topic breakdown.
  - For failed runs, a red banner with the node that crashed and the
    error message.
- **Pipeline**: per-node count stats (ingest, embed, dedupe, select,
  draft, critique, editorial, puzzles).
- **Report tab**: the optional per-article workflow report
  (`issues/{slug}/workflow-report.html`). It iframes the live file, shows
  whether it is partial or final, and offers an HTML download. With the
  config switch off, the tab explains how to enable it.
- **Drops tab**: every candidate that didn't make it into the paper, with
  stage (select quality filter, intra-dedup LLM, cross-issue similarity,
  etc.), a short reason (written by the LLM for select and dedup stages,
  synthesized otherwise), and an inline snapshot of title, source, URL,
  and summary so the entry stays readable even if the `news_items` row is
  later garbage-collected.
- **Timings** and per-stage LLM call stats (count, total duration, retries,
  timeouts, parse errors, exhausted).
- **Logs tab**: filtered tail of the backend's log stream for this
  specific run. Terminal-style panel, auto-scrolls while running,
  updates every 2 s. Drops uvicorn access noise, dashboard polls, and
  `httpx` 200-OK spam. Keeps `app.*` INFO and anything at WARNING+.
  Buffer is in-memory and TTLs 10 minutes after the run finishes; older
  issues render "No logs recorded for this run".
- **Run info**: model and endpoint used, embedding model, all tunables
  (`SELECT_BATCH_SIZE`, `SELECT_MAX_KEPT_PER_BATCH`,
  `SELECT_INTRA_SIM_THRESHOLD`, `DEDUP_SIM_THRESHOLD`, timeouts, retries),
  start/finish timestamps.

**Live updates.** The dashboard polls the details endpoint every 2 s
while the issue is `running` or `pending` and stops polling once it
hits `ready`, `failed`, or `drafted`. Logs tab polls the log endpoint on
the same cadence. No SSE or WebSocket: plain polling keeps it simple and
single-user-friendly.

The raw data lives in `issues.drafts_json.meta`. See
[architecture.md](architecture.md#metadata-captured-per-run) for the full
schema. Older issues (generated before the metadata additions) render
"Not available for this run" rather than crashing; no migration needed.

## Debug endpoints

All under `/api/debug/` on the backend (port 8000). They're diagnostic,
not part of the public API contract.

| Endpoint                                 | Purpose                                                              |
| ---------------------------------------- | -------------------------------------------------------------------- |
| `GET /api/debug/scheduler`               | List all APScheduler jobs with their next-fire times.                |
| `POST /api/debug/poll-rss`               | Run the global RSS poll right now; returns per-feed counts.          |
| `GET /api/debug/searxng?q=...&time_range=day|week|month` | Proxy a SearXNG query; see what results the agent would see.         |
| `POST /api/debug/embed {"texts":[...]}`  | Embed a test string; returns dim + first 5 floats. Useful for verifying Ollama is reachable. |
| `POST /api/debug/dedup {"window_days":14}` | Run dedup against the most-recent news_items; returns decision trace. Good for tuning `DEDUP_SIM_THRESHOLD`. |
| `POST /api/debug/agent {"config_id":...}` | Run the full agent pipeline inline (blocks the request). Useful for tests; don't use in production-ish flows. |

## Day-2 tasks

### Pause a config without deleting

Toggle "Active" off in the UI, or:

```bash
docker exec newspaper-db-1 psql -U np -d newspaper -c \
  "UPDATE newspaper_configs SET active = false WHERE name = 'daily-crosswire';"
```

The scheduler job is removed. Re-enable to re-register.

### Force re-render of an existing issue

`render_issue(issue_id)` is idempotent. It wipes old `story_log` rows for
the issue, writes fresh ones, re-copies the stylesheet, and regenerates
HTML. From inside the container:

```bash
docker compose exec -T backend python -c "
import asyncio, uuid
from app.services.renderer import render_issue
asyncio.run(render_issue(uuid.UUID('ISSUE_ID_HERE')))
print('rendered')
"
```

Useful after editing the Jinja template or style.css.

### Regenerate puzzles only (same seed, same output)

Puzzles are deterministic from `drafts_json.puzzles_meta.seed`. Re-render
(above) rebuilds them from the persisted seed. If you want *different*
puzzles, clear that seed first and the renderer will re-derive from the
issue and config ids:

```sql
UPDATE issues
SET drafts_json = jsonb_set(drafts_json, '{puzzles_meta,seed}', 'null'::jsonb)
WHERE slug = '2026-04-18-daily-crosswire';
```

### Swap LLM provider mid-stream

Edit `.env`, then:

```bash
docker compose up -d --force-recreate backend
```

`restart` is not enough: it re-reads code but not `env_file`. New runs
use the new provider; past issues and `story_log` embeddings stay as they
were (if you changed `EMBEDDING_DIM`, see
[setup.md](setup.md#changing-the-embedding-dimension)).

### Clear old news_items to save space

News items accumulate forever. To keep only the last 30 days:

```sql
DELETE FROM news_items
WHERE fetched_at < now() - interval '30 days'
  AND id NOT IN (SELECT news_item_id FROM story_log);
```

The `NOT IN` clause preserves items that appeared in a published issue;
losing those would break dedup's URL path.

### Wipe everything safely

You want a fresh start but not to nuke your configs:

```sql
-- Keep configs/topics/feeds; clear everything else.
TRUNCATE story_log, issues, news_items CASCADE;
```

Or full reset (also drops your configs):

```bash
docker compose down -v    # removes the db_data volume
docker compose up -d      # migrations re-run on first boot
```

## Troubleshooting

### Agent run says "failed" with no drafts

Open `/api/issues/{id}` and look at `warnings`. Common causes:

- **LLM returned invalid JSON** (`OutputParserException`): your model's
  structured-output support is weak. Try a larger model or a different
  provider.
- **`OPENAI_API_KEY is not configured`**: you hit the placeholder value.
  Update `.env`, `docker compose up -d --force-recreate backend`.
- **Rate-limited** (`openai.RateLimitError`): your provider is throttling.
  Wait or switch providers.
- **Everything was dropped as junk**: the LLM's select pass decided no
  story was substantive. The agent records `status = failed` if no drafts
  were produced. Relax the quality filter (edit the `SELECT_INSTRUCTION`
  prompt in `backend/app/agent/prompts.py`) or use a different feed.

### "unembedded: N" in meta

Embeddings failed for N items. Dedup quietly fell back to URL + exact-title
only for that run. Diagnose:

```bash
curl -X POST http://localhost:8000/api/debug/embed \
  -H "Content-Type: application/json" \
  -d '{"texts":["test"]}'
```

If that returns 503, your embedding endpoint is unreachable. For local
Ollama, check `ollama serve` is running. For hosted, check your API key.

### Scheduler never fires my config

```bash
curl -s http://localhost:8000/api/debug/scheduler | python3 -m json.tool
```

Look for a `config-{id}` entry. Missing usually means the config is
inactive. If present but `next_run_time` is far in the future, check the
`schedule_kind` and the weekdays/run_hour. `run_hour` is always UTC and
now applies to interval schedules too: the N-day cycle is anchored to
the next occurrence of that UTC hour.

### "Generate now" seems to hang

The UI polls every 3s once the issue is pending/running. On an Apple M4
Pro with 48 GB unified RAM the backend takes 15-30 minutes depending on
LLM latency. If it has been running notably longer than that:

```bash
docker compose logs backend --tail 200 | grep -E "agent run:|agent graph failed"
```

Look for `agent run: issue=...` when it started and `agent graph failed`
plus a traceback if something blew up.

### RSS feed says "bozo: True"

feedparser couldn't parse the response. The backend logs the feed URL and
the bozo exception. Common causes:

- The site returns HTML instead of RSS (check the URL in a browser).
- The feed requires a cookie or auth (feedparser doesn't send any).
- The site blocks our user-agent. If you see `status: 403`, edit
  `USER_AGENT` in `backend/app/services/rss.py` to a more common one.

### "duplicate key value violates unique constraint ix_news_items_url"

Shouldn't happen in current code: `ingest_search` uses
`INSERT ... ON CONFLICT DO NOTHING`. If you see it somewhere else
(custom script, old migration), add the same protection. Never let two
code paths bypass `news_items.url` uniqueness.

### Port already in use (5433, 8890, 8000, 5173)

Something else is bound to the port. `lsof -iTCP:<port> -sTCP:LISTEN` finds
it. Stop that, or edit the `ports:` section in `docker-compose.yml` to
pick different host ports. Inside-compose ports don't need to change.

### Everything is slow

The agent does a lot of synchronous work. First-run timings on an Apple
M4 Pro with 48 GB unified RAM running qwen3:14b locally:

| Step                                           | Typical duration |
| ---------------------------------------------- | ---------------- |
| RSS polling (10-20 feeds)                      | 2-5 s            |
| SearXNG queries                                | 1-3 s / topic    |
| Embedding 20 items (Ollama local)              | 5-15 s           |
| LLM `select` (per topic, ~1 round-trip)        | 20-90 s          |
| LLM `draft` (per story, ~1 round-trip)         | 30-120 s         |
| Puzzle generation (CPU-bound, all verifiers)   | 1-5 s            |
| Render + copy CSS                              | < 1 s            |

Hosted providers cut the LLM rows dramatically. If drafts dominate your
wall clock, use a faster model. If embeddings dominate, you may be
re-embedding every run (shouldn't be; `embed` only fills in missing
vectors). Check `news_items.embedding IS NULL` counts over time; they
should trend to zero.

### Newsletter delivery

Delivery state for a run lives in `meta.notify` on the issue's
`drafts_json`. The Run info tab of the Details dashboard renders it as a
"Delivery" card.

Expected statuses and what they mean:

- `ok`: all recipients received the PDF.
- `partial`: at least one recipient failed; `per_recipient` lists each
  outcome with the SMTP error.
- `skipped`: either `config.newsletter_emails` is empty (reason "no
  subscribers") or the issue isn't ready (reason "issue not ready").
- `misconfigured`: SMTP env vars aren't set; notify didn't attempt a send.
- `failed`: PDF generation or SMTP transport threw before per-recipient
  loop; `error` has the exception string.

Common failure modes:

- **Connection refused / timeout**: SMTP host unreachable from the
  backend container. Check `SMTP_HOST` resolves inside the container
  (`docker compose exec backend getent hosts <host>`). For a MailHog
  running on the Mac host, use `host.docker.internal`, not `localhost`.
- **Auth failure**: for Gmail, confirm you're using an app password, not
  the account password, and that 2FA is on. For Fastmail, confirm the app
  password isn't limited by folder permissions.
- **TLS handshake**: `SMTP_USE_TLS=true` issues STARTTLS; MailHog and
  some local servers don't support it. Set `SMTP_USE_TLS=false` for
  plain-text local testing.
- **Rate limits**: Gmail caps at ~500 recipients/day; hitting it returns
  a 421/454 response and per-recipient `failed` outcomes. Space out
  runs or use a provider without such caps.
- **Chromium fails to launch inside the container**: happens if the
  Dockerfile `playwright install chromium --with-deps` step was skipped
  or a newer image layer cached before that line. Rebuild with
  `docker compose build --no-cache backend`.

### Ad-hoc send

`POST /api/issues/{id}/send` with `{"emails": [...]}` is exposed via the
Send button on the preview page and the Run info tab. Response shape
matches `meta.notify.per_recipient`. Returns:

- `200` with `{sent, failed, per_recipient}` on happy + partial paths.
- `404` if the issue id doesn't exist.
- `409` if the issue isn't in `ready` state (can't send a failed or
  in-flight issue).
- `422` for empty or invalid email lists (caught by Pydantic
  `EmailStr` validation).
- `503` when `SMTP_HOST`/`SMTP_FROM` are blank.
- `500` if PDF generation or SMTP transport raises.

## What's not here yet

Open items deferred past the current scope:

- Solutions page (puzzle answer key).
- Multi-user or auth.
- Production deploy target (beyond docker-compose).
- Per-source trust weighting beyond `preferred_sources`.
- Recorded-response tests for CI (the current suite needs a live LLM and
  Ollama).
