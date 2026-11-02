"""Pure helpers for the select node.

Kept dependency-free so the cap rules can be unit-tested without importing
the agent graph, database models, or LLM clients.
"""

from typing import Any, Optional


def effective_topic_cap(
    topic: Optional[Any],
    config: Optional[Any],
) -> tuple[Optional[int], str]:
    """Resolve the per-topic story cap for a select run.

    A topic-level ``Topic.max_stories`` wins; otherwise the config-wide
    ``NewspaperConfig.max_stories_per_topic`` default applies. Returns
    ``(cap, source)`` where source is ``"topic"``, ``"config"``, or ``""``
    when no cap is configured.
    """
    if topic is not None and getattr(topic, "max_stories", None):
        return topic.max_stories, "topic"
    if config is not None and getattr(config, "max_stories_per_topic", None):
        return config.max_stories_per_topic, "config"
    return None, ""


def cap_reason(cap: int, source: str) -> str:
    """Human-readable drop reason for the cap that trimmed an item."""
    if source == "topic":
        return f"exceeded Topic.max_stories={cap}"
    return f"exceeded config.max_stories_per_topic={cap}"
