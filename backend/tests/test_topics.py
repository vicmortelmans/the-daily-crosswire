"""Topic ordering regression tests.

Run: docker compose exec backend pytest tests/test_topics.py -v

Topics are ordered by creation time, not heap order. PostgreSQL rewrites an
updated row at the end of the heap, so before the created_at ordering an
edited topic jumped to the bottom of the config page.
"""

import uuid


def test_topic_order_is_stable_after_update(client, clean_db):
    suffix = uuid.uuid4().hex[:8]
    cfg = client.post(
        "/api/configs",
        json={
            "name": f"__test_topic_order_{suffix}",
            "schedule_kind": "interval",
            "frequency_days": 1,
        },
    ).json()

    topic_ids = []
    for name in ("First topic", "Second topic", "Third topic"):
        resp = client.post(f"/api/configs/{cfg['id']}/topics", json={"name": name})
        assert resp.status_code == 201, resp.text
        topic_ids.append(resp.json()["id"])

    before = [t["id"] for t in client.get(f"/api/configs/{cfg['id']}").json()["topics"]]
    assert before == topic_ids

    # Updating the first topic must not move it (Postgres heap order would).
    resp = client.patch(
        f"/api/topics/{topic_ids[0]}",
        json={"search_hints": "edited", "max_stories": 2},
    )
    assert resp.status_code == 200, resp.text

    after = [t["id"] for t in client.get(f"/api/configs/{cfg['id']}").json()["topics"]]
    assert after == topic_ids
