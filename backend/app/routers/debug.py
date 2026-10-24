import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from app.agent.runner import run_generation
from app.agent.worker import agent_worker
from app.db import get_session
from app.models import NewsItem
from app.scheduler import RSS_JOB_ID, scheduler
from app.services import dedup as dedup_service
from app.services import embeddings, searxng
from app.services.rss import poll_all_feeds

router = APIRouter(prefix="/api/debug", tags=["debug"])


@router.post("/poll-rss")
def manual_poll_rss():
    return poll_all_feeds()


@router.get("/scheduler")
def scheduler_status():
    jobs = []
    for job in scheduler.get_jobs():
        jobs.append(
            {
                "id": job.id,
                "trigger": str(job.trigger),
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
            }
        )
    return {"running": scheduler.running, "jobs": jobs, "rss_job_id": RSS_JOB_ID}


@router.get("/searxng")
async def debug_searxng(q: str, time_range: Optional[str] = None, max_results: int = 5):
    try:
        return await searxng.search(q, time_range=time_range, max_results=max_results)
    except searxng.SearXNGError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


class EmbedRequest(BaseModel):
    texts: list[str]


@router.post("/embed")
async def debug_embed(payload: EmbedRequest):
    try:
        # Embedding clients are bound to the agent worker loop; never call
        # embeddings.embed() directly on the API loop or the cached httpx
        # pool will be reused from a different loop.
        vectors = await agent_worker.run(
            embeddings.embed(payload.texts), serialize=False
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return {
        "count": len(vectors),
        "dim": len(vectors[0]) if vectors else 0,
        "preview": vectors[0][:5] if vectors else [],
    }


class DedupRequest(BaseModel):
    window_days: int = 14
    sim_threshold: Optional[float] = None
    limit: int = 100


@router.post("/dedup")
def debug_dedup(
    payload: DedupRequest,
    session: Session = Depends(get_session),
):
    """Run dedup on the most recent N news_items against story_log. Useful for
    inspecting what would be dropped in a generation run without actually
    running the agent."""
    candidates = session.exec(
        select(NewsItem).order_by(NewsItem.fetched_at.desc()).limit(payload.limit)
    ).all()

    kept, decisions = dedup_service.filter_fresh(
        session,
        candidates,
        window_days=payload.window_days,
        sim_threshold=payload.sim_threshold,
    )

    breakdown: dict[str, int] = {
        "kept": 0,
        "dropped_url": 0,
        "dropped_title": 0,
        "dropped_similarity": 0,
    }
    for d in decisions:
        breakdown[d.decision] = breakdown.get(d.decision, 0) + 1

    return {
        "window_days": payload.window_days,
        "threshold": payload.sim_threshold if payload.sim_threshold is not None else None,
        "total_candidates": len(candidates),
        "kept": len(kept),
        "breakdown": breakdown,
        "decisions": [d.to_dict() for d in decisions],
    }


class AgentRunRequest(BaseModel):
    config_id: uuid.UUID


@router.post("/agent")
async def debug_run_agent(payload: AgentRunRequest):
    try:
        # Keep every agent run on the worker loop that owns the cached LLM
        # and embedding clients.
        return await agent_worker.run(run_generation(payload.config_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
