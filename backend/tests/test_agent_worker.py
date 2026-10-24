"""Tests for the dedicated agent worker loop.

These run in-process and do not need the live stack. They pin the invariant
that caused ``RuntimeError: Event loop is closed``: every agent run must use
the *same* long-lived loop, never a fresh ``asyncio.run()`` loop per job.
"""

import asyncio
import threading

import pytest

from app.agent.runner import run_generation_job, run_issue_job
from app.agent.worker import agent_worker


@pytest.fixture
def worker():
    agent_worker.start()
    try:
        yield agent_worker
    finally:
        agent_worker.shutdown()


def test_consecutive_runs_share_one_loop(worker):
    """Two sequential runs must see the same loop and worker thread.

    This is the regression guard for the cached-client bug: a new loop per
    run left httpx pooled connections attached to a closed loop. The fake
    cached client below touches its creating loop on every call, which is
    exactly what the httpx pool does when it prunes a keep-alive socket.
    """
    cached_client = {}

    async def use_cached_client():
        loop = asyncio.get_running_loop()
        client = cached_client.get("client")
        if client is None:
            client = cached_client["client"] = {"loop": loop}
        if client["loop"].is_closed():
            raise RuntimeError("Event loop is closed")
        # Any loop-bound primitive call fails on a closed loop.
        client["loop"].create_future().cancel()
        return id(loop), threading.current_thread().name

    first = worker.run_blocking(use_cached_client())
    second = worker.run_blocking(use_cached_client())

    assert first == second
    assert first[1] == "agent-worker"


def test_runs_are_serialized(worker):
    """A second run waits for the first, so clients and state never interleave."""

    state = {"active": 0, "max_active": 0}

    async def measure():
        state["active"] += 1
        state["max_active"] = max(state["max_active"], state["active"])
        await asyncio.sleep(0.05)
        state["active"] -= 1
        return True

    first = worker.submit(measure())
    second = worker.submit(measure())
    assert first.result(timeout=5) is True
    assert second.result(timeout=5) is True
    assert state["max_active"] == 1


def test_run_from_async_context(worker):
    """FastAPI/APScheduler callers await worker.run() without blocking."""

    async def value():
        return asyncio.get_running_loop().is_running()

    async def main():
        return await worker.run(value())

    assert asyncio.run(main()) is True


def test_worker_restarts_after_shutdown(worker):
    async def loop_id():
        return id(asyncio.get_running_loop())

    first_loop = worker.run_blocking(loop_id())
    worker.shutdown()
    worker.start()
    second_loop = worker.run_blocking(loop_id())

    assert first_loop != second_loop
    assert worker.running


def test_scheduler_entry_points_are_coroutines():
    """APScheduler's AsyncIOExecutor only awaits coroutine functions; sync
    wrappers would land in its thread pool and reintroduce per-job loops."""
    assert asyncio.iscoroutinefunction(run_generation_job)
    assert asyncio.iscoroutinefunction(run_issue_job)
