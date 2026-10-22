import asyncio
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest
from langgraph.graph import END
from sqlmodel import Session

from app.agent import nodes
from app.agent.graph import _build_graph
from app.agent.state import AgentState
from app.config import settings
from app.db import engine
from app.models import Issue, NewspaperConfig


def _create_ready_issue(slug: str = "2026-10-18-dat-zal-wel") -> tuple[uuid.UUID, uuid.UUID]:
    with Session(engine) as s:
        cfg = NewspaperConfig(name="Test Paper", schedule_kind="interval", frequency_days=1)
        s.add(cfg)
        s.commit()
        s.refresh(cfg)

        issue = Issue(
            config_id=cfg.id,
            slug=slug,
            issue_date=date(2026, 10, 18),
            status="ready",
        )
        s.add(issue)
        s.commit()
        s.refresh(issue)
        return cfg.id, issue.id


def test_print_skipped_when_no_printer_configured(monkeypatch):
    monkeypatch.setattr(settings, "printer", "")
    state: AgentState = {
        "issue_id": str(uuid.uuid4()),
        "config_id": str(uuid.uuid4()),
        "meta": {},
    }

    result = asyncio.run(nodes.print_issue(state))
    print_meta = result.get("meta", {}).get("print", {})
    assert print_meta.get("status") == "skipped"
    assert print_meta.get("reason") == "no printer configured"


def test_print_skipped_when_issue_not_ready(monkeypatch):
    monkeypatch.setattr(settings, "printer", "Epson-WF-7210")
    with Session(engine) as s:
        cfg = NewspaperConfig(name="Test Paper", schedule_kind="interval", frequency_days=1)
        s.add(cfg)
        s.commit()
        s.refresh(cfg)

        issue = Issue(
            config_id=cfg.id,
            slug="test-pending",
            issue_date=date(2026, 10, 18),
            status="pending",
        )
        s.add(issue)
        s.commit()
        s.refresh(issue)
        cfg_id, issue_id = cfg.id, issue.id

    state: AgentState = {
        "issue_id": str(issue_id),
        "config_id": str(cfg_id),
        "meta": {},
    }

    result = asyncio.run(nodes.print_issue(state))
    print_meta = result.get("meta", {}).get("print", {})
    assert print_meta.get("status") == "skipped"
    assert print_meta.get("reason") == "issue not ready"


def test_print_failed_when_pdf_generation_fails(monkeypatch):
    monkeypatch.setattr(settings, "printer", "Epson-WF-7210")
    cfg_id, issue_id = _create_ready_issue("test-pdf-fail")

    state: AgentState = {
        "issue_id": str(issue_id),
        "config_id": str(cfg_id),
        "meta": {},
    }

    with patch("app.services.pdf.build_issue_pdf", side_effect=FileNotFoundError("missing html")):
        result = asyncio.run(nodes.print_issue(state))

    print_meta = result.get("meta", {}).get("print", {})
    assert print_meta.get("status") == "failed"
    assert "FileNotFoundError" in print_meta.get("error", "")
    assert any("print PDF generation failed" in w for w in result.get("warnings", []))


def test_print_failed_when_lp_command_fails(monkeypatch):
    monkeypatch.setattr(settings, "printer", "Epson-WF-7210")
    cfg_id, issue_id = _create_ready_issue("test-lp-fail")

    state: AgentState = {
        "issue_id": str(issue_id),
        "config_id": str(cfg_id),
        "meta": {},
    }

    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.communicate = AsyncMock(
        return_value=(b"", b"lp: Error - The printer or class was not found.")
    )

    with (
        patch("app.services.pdf.build_issue_pdf", AsyncMock(return_value=b"%PDF-1.4 mock")),
        patch("asyncio.create_subprocess_exec", AsyncMock(return_value=mock_proc)),
    ):
        result = asyncio.run(nodes.print_issue(state))

    print_meta = result.get("meta", {}).get("print", {})
    assert print_meta.get("status") == "failed"
    assert "The printer or class was not found." in print_meta.get("error", "")
    assert any("print failed" in w for w in result.get("warnings", []))


def test_print_success_calls_lp_with_exact_arguments(monkeypatch):
    monkeypatch.setattr(settings, "printer", "Epson-WF-7210")
    monkeypatch.setattr(settings, "cups_host", "host.docker.internal")
    cfg_id, issue_id = _create_ready_issue("2026-10-18-dat-zal-wel")

    state: AgentState = {
        "issue_id": str(issue_id),
        "config_id": str(cfg_id),
        "meta": {},
    }

    captured_calls = []

    async def mock_exec(*cmd, cwd=None, env=None, stdout=None, stderr=None):
        # Verify file exists in cwd and contains the expected bytes at execution time
        pdf_file = Path(cwd) / "2026-10-18-dat-zal-wel.pdf"
        assert pdf_file.exists()
        assert pdf_file.read_bytes() == b"%PDF-1.4 mock data"

        captured_calls.append({"cmd": list(cmd), "cwd": cwd, "env": env})
        proc = MagicMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(
            return_value=(b"request id is Epson-WF-7210-42 (1 file(s))\n", b"")
        )
        return proc

    with (
        patch("app.services.pdf.build_issue_pdf", AsyncMock(return_value=b"%PDF-1.4 mock data")),
        patch("asyncio.create_subprocess_exec", side_effect=mock_exec),
    ):
        result = asyncio.run(nodes.print_issue(state))

    assert len(captured_calls) == 1
    call = captured_calls[0]
    expected_cmd = [
        "lp",
        "-h",
        "host.docker.internal",
        "-d",
        "Epson-WF-7210",
        "-o",
        "PageSize=A3",
        "-o",
        "MediaType=PLAIN_NORMAL",
        "-o",
        "Duplex=DuplexNoTumble",
        "-o",
        "InputSlot=Auto",
        "2026-10-18-dat-zal-wel.pdf",
    ]
    assert call["cmd"] == expected_cmd
    assert call["env"].get("CUPS_SERVER") == "host.docker.internal"

    print_meta = result.get("meta", {}).get("print", {})
    assert print_meta.get("status") == "ok"
    assert print_meta.get("printer") == "Epson-WF-7210"
    assert "request id is Epson-WF-7210-42" in print_meta.get("output", "")


def test_graph_structure():
    graph = _build_graph()

    # The graph has the print node
    assert "print" in graph.nodes

    # The edges connect notify -> print -> END
    edges = list(graph.builder.edges)
    assert ("notify", "print") in edges
    assert ("print", END) in edges

