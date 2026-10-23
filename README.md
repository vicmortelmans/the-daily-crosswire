# The Daily Crosswire

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-agent-1C3C3C)](https://langchain-ai.github.io/langgraph/)
[![Node](https://img.shields.io/badge/node-22-339933?logo=node.js&logoColor=white)](https://nodejs.org/)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-8-646CFF?logo=vite&logoColor=white)](https://vitejs.dev/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20%2B%20pgvector-336791?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![Docker Compose](https://img.shields.io/badge/Docker%20Compose-required-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![Status](https://img.shields.io/badge/status-alpha-orange)](#)

A self-hosted web app that generates a small, print-ready newspaper on a
schedule. You define the topics and sources; an LLM agent ingests RSS
feeds and web search results, deduplicates against past runs, writes tight
cause-and-effect stories, assembles a themed puzzles page, and renders an
HTML issue you can open in the browser or save as a PDF via Cmd+P. If you
subscribe email addresses to the config, every successful run also mails
a PDF to the list automatically.


## What it does

- **You configure newspapers.** Each has a schedule (every N days, or
  specific weekdays), a set of topics, per-topic preferred sources, free-text
  search hints, optional RSS feeds, and an **output language** (EN / ES /
  FR / IT / PT / DE). The agent translates multilingual sources into the
  chosen language at draft time.
- **On schedule (or manual trigger), the agent runs:**
  1. Polls RSS feeds for every topic that has any.
  2. Falls back to SearXNG web search for topics with no feeds or too few
     fresh items.
  3. Embeds candidate stories with your chosen embedding model (defaults to
     `nomic-embed-text` via local Ollama, but any OpenAI-compatible endpoint
     works).
  4. Deduplicates against recent published issues by URL, normalized
     title, and pgvector cosine similarity.
  5. **Quality filter with batching:** candidates are chunked
     (`SELECT_BATCH_SIZE=10`) and each chunk is shown to the LLM with a
     hard cap on how many it may keep (`SELECT_MAX_KEPT_PER_BATCH=5`).
     The prompt carries two rubrics (entertainment vs. local/city news)
     and keeps the near-term concrete stories, drops trailers/DLC/merch/
     aggregator handles.
  6. **Intra-issue dedup (two passes):** an LLM pass clusters same-event
     duplicates across batches and picks the most specific item per
     cluster as the survivor; an embedding safety net then runs a cheap
     cosine collapse on title+summary above `SELECT_INTRA_SIM_THRESHOLD`.
  7. Drafts 100-150 word summaries per story, plus one short editorial.
  8. Generates a themed puzzles page (sudoku, criss-cross, word search,
     maze, tic-tac-toe, cryptogram). Every puzzle is verified solvable
     before emission.
  9. Renders the final HTML at `issues/{slug}/index.html` with a print-ready
     stylesheet, localized to the config's chosen language (date format,
     masthead strings, editorial label, etc.).
  10. Mails a PDF of the rendered issue to `config.newsletter_emails` via
      SMTP. Playwright renders the same HTML through headless Chromium so
      the attachment matches what Cmd+P would save. Runs per recipient so
      per-recipient outcomes are recorded in `meta.notify`. Skipped cleanly
      when the list is empty or SMTP is not configured.
- **Every drop is traced.** Each candidate that didn't make it into the
  paper (dropped by cross-issue dedup, quality filter, intra-issue dedup,
  or per-topic cap) lands in `meta.drop_trace` with the stage, a short
  reason (the LLM writes its own for select and dedup), and a denormalized
  snapshot of the source so it stays readable even if the source news_items
  row is garbage-collected later. Drops include the item's `published_at`
  and ingestion origin (RSS feed URL or SearXNG query) when available.
- **Optional per-article workflow report.** Each config can emit a standalone
  HTML report while the agent runs. It is organized by retrieved link and
  shows the full chain: RSS/search input, embedding, cross-issue dedup, full
  article fetch, quality-filter keep/drop motivation, intra-issue dedup,
  topic cap, draft input/output, and critique rewrites. The report and its
  JSON event ledger live in `issues/{slug}/`; no trace data is stored in
  Postgres. Disable it per config with **Generate workflow report**.
- **You browse.** The web UI lists past issues with date/config/status
  filters, previews each one in an iframe, and has:
  - a one-click **Print / Save as PDF**;
  - an ad-hoc **Send this issue** form next to the print button for
    emailing a ready issue to a one-off recipient list;
  - a **Details dashboard** per issue with seven tabs: Overview, Pipeline,
    Drops, Report, Timings, Logs, Run info. **Live while the run is in
    progress:** the dashboard shows the current node, elapsed time per
    step, completed nodes as green checkmarks, and a filtered log tail.
    Failed issues route here too with a banner pointing at the node that
    crashed and the traceback. Older issues without the newer fields
    degrade gracefully with "Not available for this run" rather than
    crashing.

## Quick start

```bash
cp .env.example .env
# edit .env: set OPENAI_API_KEY, and EMBEDDING_* if you want a local model
docker compose up -d --build

# open the UI
open http://localhost:5173
```

Create a config, add a topic with an RSS feed, hit Generate now. The
agent runs end-to-end from one click, with no need to wait for the scheduler.

## Architecture at a glance

```
┌────────────┐      ┌──────────────────────────────────────┐      ┌────────────┐
│  React UI  │─────▶│ FastAPI                               │◀───▶│ Postgres   │
│  (Vite)    │      │ ├── REST API (configs, issues)        │      │ + pgvector │
│            │◀─────│ ├── APScheduler (per-config triggers) │      └────────────┘
└────────────┘      │ ├── RSS ingester + dedup + renderer   │
      ▲             │ ├── LangGraph agent (12 nodes)        │─────▶ SearXNG
      │             │ ├── Playwright PDF + SMTP delivery    │       (self-hosted)
      │             │ └── /issues/{slug}/ static mount      │
      │             └──────────────────────────────────────┘             │
      └── iframes /issues/{slug}/index.html for preview                  │
                                                                         ▼
                                                          ┌────────────────────┐
                                                          │ OpenAI-compatible  │
                                                          │ LLM + embeddings   │
                                                          │ (OpenAI, Ollama,   │
                                                          │  OpenRouter, ...)  │
                                                          └────────────────────┘
```

Everything runs via `docker compose` on your laptop. The only external
things you supply are a chat LLM endpoint (API key) and optionally an
embedding endpoint.

## Host ports (non-standard)

| Service   | Host port | Why                                                              |
| --------- | --------- | ---------------------------------------------------------------- |
| backend   | 8000      |                                                                  |
| ui        | 5173      |                                                                  |
| searxng   | **8890**  | 8080/8888 collided with other things running on dev machines     |
| postgres  | **5433**  | 5432 collided with a local pg                                    |

Inside the compose network, services still use their default container
ports (`db:5432`, `searxng:8080`). Override only if you need to.

## Repo layout

```
.
├── backend/           FastAPI + LangGraph + puzzle generator
│   ├── app/
│   │   ├── agent/          LangGraph nodes, prompts, runner, LLM transport
│   │   ├── puzzles/        Puzzle generators + verifiers
│   │   ├── routers/        REST endpoints
│   │   ├── services/       rss, searxng, embeddings, dedup, renderer
│   │   ├── templates/      Jinja template + print stylesheet
│   │   └── utils/
│   ├── alembic/            DB migrations
│   ├── scripts/            Labeled LLM benchmarks (latency + quality)
│   └── tests/              pytest suite
├── ui/                React + Vite + TanStack Query + Tailwind
├── searxng/           Local SearXNG config (JSON mode enabled)
├── issues/            Generated newspapers (git-ignored)
├── docs/              User documentation (see below)
└── docker-compose.yml
```

## Where to read more

- [docs/setup.md](docs/setup.md): prerequisites, env vars, first-time
  bring-up, swapping LLM providers, troubleshooting.
- [docs/architecture.md](docs/architecture.md): how the pipeline works
  end-to-end: ingestion, dedup, agent graph, renderer, scheduler.
- [docs/operations.md](docs/operations.md): day-to-day tasks: running
  tests, CLIs, debug endpoints, swapping models, regenerating issues,
  clearing data safely.
- [docs/voice.md](docs/voice.md): content and tone rules the agent follows.

## Non-negotiables

These are baked in and earned their place through testing + use:

- **Single user, no auth.** Keeps the whole app simple.
- **LLM via OpenAI-compatible endpoint, not hardcoded to a specific
  provider.** Swap freely between OpenAI, OpenRouter, Groq, local Ollama,
  local vLLM, LM Studio, etc.
- **Every puzzle is proven solvable before HTML is emitted.** If the
  generator can't prove it, the run fails loudly.
- **No hype words, no superlatives, no exclamation marks, no emoji in
  any user-facing copy.** The voice is straight news.
- **Cmd+P is the primary PDF step for interactive users.** A headless
  Chromium via Playwright exists in one place only:
  [backend/app/services/pdf.py](backend/app/services/pdf.py), used by the
  newsletter delivery path so the emailed attachment matches what
  Cmd+P would save.
- **Newsletter delivery via SMTP.** Per-config recipient list fires
  automatically on every successful render; the preview page also
  exposes an ad-hoc send to a one-off list.

## License

Apache License 2.0. See [LICENSE](LICENSE).
