"""pytest integration: a ``tape`` fixture that records each test's run.

Loaded automatically when pytest and tapedeck are both installed (the
``pytest11`` entry point); it stays dormant until a test requests the
fixture::

    def test_agent_replies(tape):
        reply = my_agent(tape, "hello")     # executed live, traced
        assert "3.50" in reply

Every test that asks for ``tape`` leaves a run under ``.tapedeck/runs/``:
``tapedeck ls`` lists them, ``tapedeck diff`` compares two test runs, and a
recording made here can be replayed offline anywhere -- including back
inside pytest with ``tapedeck.play`` (see the README's CI section).
"""
from __future__ import annotations

import re

import pytest

import tapedeck


def _safe_name(nodeid: str) -> str:
    name = nodeid.replace("::", "-").replace("/", "-").replace("\\", "-")
    return re.sub(r"[^A-Za-z0-9._-]", "-", name)[:80]


@pytest.fixture
def tape(request):
    """A recording tape scoped to one test; saved even if the test fails."""
    tape_ = tapedeck.record("pytest-" + _safe_name(request.node.nodeid))
    yield tape_
    tapedeck.save(tape_)
