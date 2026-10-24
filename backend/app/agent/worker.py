"""Dedicated, long-lived event loop for agent runs.

Why this exists
---------------
LLM and embedding clients (langchain ``ChatOpenAI``, ``AsyncOpenAI`` and the
httpx connection pools they own) bind their sockets to the event loop that
created them. Generation used to run through ``asyncio.run()`` inside an
APScheduler worker thread, so every run got a fresh loop and that loop was
closed when the coroutine finished. The process-wide cached clients then kept
keep-alive connections from a loop that no longer existed, and the next run
failed with ``RuntimeError: Event loop is closed``.

Everything that touches those clients now runs on one loop owned by this
module. Async callers (FastAPI endpoints, APScheduler jobs) submit work with
``await agent_worker.run(coro)``; synchronous callers use
``agent_worker.run_blocking(coro)``. Runs are serialized on the worker loop so
two configs cannot interleave and share client state.

The loop lives in a daemon thread, so a hung run can never keep the process
alive after ``shutdown()``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from typing import Any, Coroutine, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_START_TIMEOUT_SECONDS = 10.0
_SHUTDOWN_TIMEOUT_SECONDS = 10.0


class AgentWorker:
    """Owns the single event loop used by the agent pipeline."""

    def __init__(self) -> None:
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        # Created on the worker loop in _run_forever, used from _serialized
        # (which also executes on the worker loop).
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        thread = self._thread
        loop = self._loop
        return (
            thread is not None
            and thread.is_alive()
            and loop is not None
            and not loop.is_closed()
        )

    def start(self) -> None:
        """Start the worker thread/loop if it is not already running."""
        with self._lock:
            if self.running:
                return
            self._ready.clear()
            self._thread = threading.Thread(
                target=self._run_forever,
                name="agent-worker",
                daemon=True,
            )
            self._thread.start()

        if not self._ready.wait(timeout=_START_TIMEOUT_SECONDS):
            raise RuntimeError(
                f"agent worker event loop did not start within {_START_TIMEOUT_SECONDS:.0f}s"
            )
        logger.info("agent worker started (thread=%s)", self._thread.name)

    def _run_forever(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._semaphore = asyncio.Semaphore(1)
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
                loop.run_until_complete(loop.shutdown_asyncgens())
            finally:
                loop.close()
                self._loop = None
                self._semaphore = None

    def _ensure_started(self) -> asyncio.AbstractEventLoop:
        if not self.running:
            self.start()
        loop = self._loop
        if loop is None or loop.is_closed():
            raise RuntimeError("agent worker event loop is not available")
        return loop

    def submit(
        self, coro: Coroutine[Any, Any, T], *, serialize: bool = True
    ) -> concurrent.futures.Future[T]:
        """Schedule ``coro`` on the worker loop.

        ``serialize=True`` (the default) holds the single-run semaphore, so
        two agent generations can never interleave. Short standalone calls
        (e.g. the debug embedding probe) pass ``serialize=False`` so they are
        not stuck behind a multi-hour generation.
        """

        async def _wrapped() -> T:
            semaphore = self._semaphore
            if semaphore is None:
                raise RuntimeError("agent worker semaphore is not initialized")
            if not serialize:
                return await coro
            async with semaphore:
                return await coro

        try:
            loop = self._ensure_started()
            future = asyncio.run_coroutine_threadsafe(_wrapped(), loop)
        except Exception:
            coro.close()
            raise
        # If shutdown cancels the wrapper before it ever awaits `coro`,
        # Python would otherwise warn that the coroutine was never awaited.
        future.add_done_callback(lambda _f: coro.close())
        return future

    def run_blocking(self, coro: Coroutine[Any, Any, T], *, serialize: bool = True) -> T:
        """Run ``coro`` on the worker loop and block until it finishes."""
        if threading.current_thread() is self._thread:
            coro.close()
            raise RuntimeError(
                "run_blocking() called from the agent worker loop; "
                "use 'await agent_worker.run(...)' instead"
            )
        return self.submit(coro, serialize=serialize).result()

    async def run(self, coro: Coroutine[Any, Any, T], *, serialize: bool = True) -> T:
        """Await ``coro`` running on the worker loop without blocking ours."""
        future = self.submit(coro, serialize=serialize)
        return await asyncio.wrap_future(future)

    def shutdown(self, timeout: float = _SHUTDOWN_TIMEOUT_SECONDS) -> None:
        """Cancel in-flight work, stop the loop, and join the worker thread."""
        with self._lock:
            loop = self._loop
            thread = self._thread
            if loop is None or thread is None:
                return

            async def _cancel_active() -> None:
                current = asyncio.current_task()
                tasks = [t for t in asyncio.all_tasks() if t is not current]
                for task in tasks:
                    task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

            try:
                asyncio.run_coroutine_threadsafe(_cancel_active(), loop).result(
                    timeout=timeout
                )
            except Exception:
                logger.warning(
                    "agent worker shutdown: could not cancel in-flight tasks",
                    exc_info=True,
                )
            loop.call_soon_threadsafe(loop.stop)

        thread.join(timeout=timeout)
        if thread.is_alive():
            # Do not clear the thread handle: start() would otherwise spawn a
            # second loop while the old one (and its bound clients) still
            # exists, which is exactly the failure mode this module prevents.
            logger.warning(
                "agent worker thread did not stop within %.0fs; not restarting it",
                timeout,
            )
            return
        self._thread = None
        self._ready.clear()
        logger.info("agent worker stopped")


agent_worker = AgentWorker()
