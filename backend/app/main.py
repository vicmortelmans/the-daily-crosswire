import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import scheduler as scheduler_module
from app.agent.worker import agent_worker
from app.config import settings
from app.routers import configs, debug, feeds, issues, topics
from app.services import log_buffer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log_buffer.install_root_handler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # The agent worker must start before any job is enqueued: it owns the
    # long-lived event loop that every cached LLM/embedding client binds to.
    agent_worker.start()
    scheduler_module.start()
    try:
        yield
    finally:
        scheduler_module.shutdown()
        agent_worker.shutdown()


app = FastAPI(title="Newspaper", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(configs.router)
app.include_router(topics.router)
app.include_router(feeds.router)
app.include_router(issues.router)
app.include_router(debug.router)

# Static mount for generated newspapers.
settings.issues_dir.mkdir(parents=True, exist_ok=True)
app.mount(
    "/issues",
    StaticFiles(directory=str(settings.issues_dir), html=True),
    name="issues",
)


@app.get("/")
def root():
    return {"service": "newspaper-backend"}


@app.get("/api/health")
def health():
    return {"status": "ok"}
