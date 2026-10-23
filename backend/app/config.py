from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://np:np@db:5432/newspaper"
    searxng_url: str = "http://searxng:8080"

    openai_api_base: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    # Optional explicit Ollama override. When OLLAMA_API_BASE and OLLAMA_MODEL
    # are BOTH set, the agent uses langchain-ollama (native tool calling via
    # /api/chat) instead of ChatOpenAI against the OpenAI-compat layer. Leave
    # both blank to stay on the OPENAI_* path (any OpenAI-compatible endpoint).
    # OLLAMA_API_BASE should NOT include /v1 (e.g. http://host.docker.internal:11434).
    ollama_api_base: str = ""
    ollama_model: str = ""
    # Per-call LLM timeout in seconds and how many retries the SDK attempts.
    # Bump higher for slow local models (llama.cpp, small Ollama) or big prompts.
    llm_timeout_seconds: int = 300
    llm_max_retries: int = 3

    # Select (quality filter) is called once per topic. When a topic has many
    # candidates after dedup, a single prompt gets huge and the call times
    # out. Instead we split candidates into chunks of this size, call the
    # LLM per chunk, and merge the kept lists. Smaller = more robust /
    # more API calls. Larger = fewer calls / risk of timeouts.
    select_batch_size: int = 10

    # Hard cap on how many items the LLM is allowed to keep PER BATCH.
    # With select_batch_size=10 and this at 5, no batch can contribute more
    # than half its items. Different from the per-topic cap on NewspaperConfig
    # which trims after all batches merge; this one bounds each chunk at the
    # source so one batch can't flood the selection.
    select_max_kept_per_batch: int = 5

    # Intra-issue near-dup collapse threshold. Separate from dedup_sim_threshold
    # (which runs cross-issue against story_log). Intra-issue dupes are same-day
    # coverage of one event across outlets; titles paraphrase heavily, so
    # this is looser (0.75) than the cross-issue strict 0.88. We also re-embed
    # title+summary (not stored title-only) for this pass to tighten clusters.
    select_intra_sim_threshold: float = 0.75

    # Embeddings use their own endpoint/key so you can mix providers
    # (e.g. local embeddings + hosted chat). If left blank they fall back
    # to the OPENAI_* values above.
    embedding_api_base: str = ""
    embedding_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    @property
    def effective_embedding_api_base(self) -> str:
        return self.embedding_api_base or self.openai_api_base

    @property
    def effective_embedding_api_key(self) -> str:
        return self.embedding_api_key or self.openai_api_key

    dedup_sim_threshold: float = 0.88
    rss_poll_interval_hours: int = 1

    # Full article extraction settings for the fetch_articles node.
    # Fetches full article bodies for candidates before selection.
    article_fetch_timeout_seconds: float = 10.0
    # Hard cap on extracted body length, applied before the text is cached in
    # news_items.article_text. Long-reads/liveblogs otherwise bloat the select
    # prompt and can time out the LLM. Also bounds the draft source text.
    article_max_characters: int = 3000
    article_fetch_concurrency: int = 5

    issues_dir: Path = Path("/app/issues")

    # Comma-separated list of origins the CORS middleware accepts.
    # Default matches the Vite dev server inside docker-compose; set
    # CORS_ALLOWED_ORIGINS to override when deploying behind a different
    # hostname.
    cors_allowed_origins: str = "http://localhost:5173"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]

    # SMTP settings for the notify node and the ad-hoc /send endpoint.
    # Leave any required field blank to disable delivery (notify records
    # status='misconfigured', ad-hoc /send returns 503). Works with any
    # plain-SMTP provider: Gmail app password, Fastmail, self-hosted,
    # MailHog for local testing.
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    smtp_use_tls: bool = True

    @property
    def smtp_configured(self) -> bool:
        return bool(self.smtp_host and self.smtp_from)

    # CUPS printer settings for the print node in the agent pipeline.
    # PRINTER in .env sets the target printer queue (e.g. Epson-WF-7210).
    # If empty, the print node is skipped.
    printer: str = ""
    cups_host: str = "host.docker.internal"

    @property
    def printer_configured(self) -> bool:
        return bool(self.printer and self.printer.strip())


settings = Settings()
