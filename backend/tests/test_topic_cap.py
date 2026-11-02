"""Per-topic story cap resolution.

The select node resolves the cap in one place: a Topic.max_stories value
wins, otherwise the config-wide NewspaperConfig.max_stories_per_topic
default applies. These are pure unit tests; no database or LLM needed.

Uses stdlib unittest so the file can run without pytest installed:
    python3 -m unittest tests.test_topic_cap -v
"""

import unittest
from types import SimpleNamespace

from app.agent.caps import cap_reason, effective_topic_cap


class EffectiveTopicCapTests(unittest.TestCase):
    def test_topic_cap_wins_over_config_default(self):
        topic = SimpleNamespace(max_stories=3)
        config = SimpleNamespace(max_stories_per_topic=10)

        self.assertEqual(effective_topic_cap(topic, config), (3, "topic"))

    def test_topic_falls_back_to_config_default(self):
        topic = SimpleNamespace(max_stories=None)
        config = SimpleNamespace(max_stories_per_topic=7)

        self.assertEqual(effective_topic_cap(topic, config), (7, "config"))

    def test_no_cap_when_neither_is_set(self):
        topic = SimpleNamespace(max_stories=None)

        self.assertEqual(
            effective_topic_cap(topic, SimpleNamespace(max_stories_per_topic=None)),
            (None, ""),
        )
        self.assertEqual(effective_topic_cap(topic, None), (None, ""))
        self.assertEqual(
            effective_topic_cap(None, SimpleNamespace(max_stories_per_topic=4)),
            (4, "config"),
        )

    def test_cap_reason_names_the_source(self):
        self.assertEqual(cap_reason(3, "topic"), "exceeded Topic.max_stories=3")
        self.assertEqual(cap_reason(5, "config"), "exceeded config.max_stories_per_topic=5")


if __name__ == "__main__":
    unittest.main()
