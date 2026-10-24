import logging
import time
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.agent import nodes
from app.agent.checkpoint import checkpoint_meta
from app.agent.state import AgentState
from app.services import log_buffer
from app.services import workflow_report
from app.utils.time import now_utc

logger = logging.getLogger(__name__)


def _timed(name: str, fn):
    """Wrap a node so its wall-clock duration lands in state.meta.timings
    and the whole meta snapshot checkpoints to the DB at node boundaries.

    At entry: record current_node + node_started_at and checkpoint so the
    UI sees which step is running.
    At exit:  record timing + append to completed_nodes and checkpoint
    again so the UI catches progress even if the next node is slow.

    Also flushes the in-memory log buffer to the issue_logs table at both
    boundaries. That way, even if _evict_stale() wipes the ring during a
    long silent window, the /logs endpoint can still serve the full
    history from the DB fallback."""

    async def wrapper(state: AgentState) -> AgentState:
        issue_id = state.get("issue_id") if isinstance(state, dict) else None
        meta_in = state.setdefault("meta", {}) if isinstance(state, dict) else {}
        if issue_id:
            meta_in["current_node"] = name
            meta_in.setdefault("node_started_at", {})[name] = now_utc().isoformat()
            checkpoint_meta(issue_id, meta_in)
            log_buffer.flush_to_db(issue_id)

        t0 = time.perf_counter()
        try:
            result = await fn(state)
        except Exception:
            # Keep whatever report has been assembled so far. Best-effort.
            workflow_report.write_if_due(state, final=False)
            raise
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        if isinstance(result, dict):
            meta = result.setdefault("meta", {})
            meta.setdefault("timings", {})[name] = elapsed_ms
            completed = meta.setdefault("completed_nodes", [])
            if name not in completed:
                completed.append(name)
            # Clear current_node when the last node (print) finishes;
            # otherwise the next node's entry wrapper will overwrite it.
            if name == "print":
                meta["current_node"] = None
            if issue_id:
                checkpoint_meta(issue_id, meta)
                log_buffer.flush_to_db(issue_id)
            # Surface non-article nodes in the report appendix.
            if name in {"puzzles", "render", "notify", "print"}:
                section = meta.get(name)
                if isinstance(section, dict):
                    status = section.get("status") or ("failed" if section.get("error") else "ok")
                    workflow_report.record_global(
                        result,
                        name,
                        status,
                        data={k: v for k, v in section.items() if k != "progress"},
                    )
            # Live report updates at every node boundary; final at the end.
            workflow_report.write_if_due(result, final=(name == "print"))
        return result

    wrapper.__name__ = f"timed_{name}"
    return wrapper


def _build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("ingest_rss", _timed("ingest_rss", nodes.ingest_rss))
    g.add_node("ingest_search", _timed("ingest_search", nodes.ingest_search))
    g.add_node("embed", _timed("embed", nodes.embed))
    g.add_node("dedupe", _timed("dedupe", nodes.dedupe))
    g.add_node("select", _timed("select", nodes.select_stories))
    g.add_node("fetch_articles", _timed("fetch_articles", nodes.fetch_articles))
    g.add_node("quote", _timed("quote", nodes.quote))
    g.add_node("draft", _timed("draft", nodes.draft))
    g.add_node("critique", _timed("critique", nodes.critique))
    g.add_node("editorial", _timed("editorial", nodes.editorial))
    g.add_node("puzzles", _timed("puzzles", nodes.puzzles))
    g.add_node("finalize", _timed("finalize", nodes.finalize))
    g.add_node("render", _timed("render", nodes.render))
    g.add_node("notify", _timed("notify", nodes.notify))
    g.add_node("print", _timed("print", nodes.print_issue))

    g.add_edge(START, "ingest_rss")
    g.add_edge("ingest_rss", "ingest_search")
    g.add_edge("ingest_search", "embed")
    g.add_edge("embed", "dedupe")
    g.add_edge("dedupe", "fetch_articles")
    g.add_edge("fetch_articles", "select")
    g.add_edge("select", "quote")
    g.add_edge("quote", "draft")
    g.add_edge("draft", "critique")
    g.add_edge("critique", "editorial")
    g.add_edge("editorial", "puzzles")
    g.add_edge("puzzles", "finalize")
    g.add_edge("finalize", "render")
    g.add_edge("render", "notify")
    g.add_edge("notify", "print")
    g.add_edge("print", END)

    return g


@lru_cache(maxsize=1)
def get_graph():
    return _build_graph().compile()
