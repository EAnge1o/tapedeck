"""Run storage: one directory per recorded run.

    .tapedeck/runs/<timestamp>-<name>/
        meta.json     run header (name, host, python, counts, totals)
        trace.jsonl   one Event per line, schema-versioned

Plain files on purpose: runs are `git`-able, `rsync`-able, `cat`-able, and
survive every future migration of every SaaS.
"""
from __future__ import annotations

import json
import platform
import re
import socket
import time
from pathlib import Path

from .session import Event, RecordingTape, SCHEMA_VERSION

DEFAULT_ROOT = Path(".tapedeck") / "runs"

_RUN_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")
_RUN_DIR_RE = re.compile(r"^\d{8}-\d{6}-(?P<name>.+)$")


def sanitize_run_name(name: str) -> str:
    cleaned = _RUN_NAME_RE.sub("-", name).strip("-") or "run"
    return cleaned[:48]


def _run_name_of(dir_name: str) -> str | None:
    """Extract the run name from '<stamp>-<name>' (the stamp holds one dash)."""
    match = _RUN_DIR_RE.match(dir_name)
    return match.group("name") if match else None


class RunStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else DEFAULT_ROOT

    def new_run_dir(self, name: str) -> Path:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = self.root / f"{stamp}-{sanitize_run_name(name)}"
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def list_runs(self) -> list[Path]:
        if not self.root.exists():
            return []
        # Newest first: by modification time, not name (names sort
        # alphabetically, which quietly lies about "latest").
        return sorted(
            (p for p in self.root.iterdir() if (p / "trace.jsonl").exists()),
            key=lambda p: (p / "trace.jsonl").stat().st_mtime,
            reverse=True,
        )

    def resolve(self, ref: str) -> Path:
        """Accept an exact run directory, a path, or a unique name prefix.

        Prefixes match either the full directory name (``20261026-``) or the
        run name embedded in it (``runA``), whichever the user knows.
        """
        candidate = Path(ref)
        if (candidate / "trace.jsonl").exists():
            return candidate
        matches = [
            p for p in self.list_runs()
            if p.name.startswith(ref) or (_run_name_of(p.name) or "").startswith(ref)
        ]
        if not matches:
            raise FileNotFoundError(
                f"no run matches {ref!r} under {self.root} "
                f"(found {len(self.list_runs())} runs)"
            )
        if len(matches) > 1:
            names = ", ".join(m.name for m in matches[:5])
            raise FileNotFoundError(f"run prefix {ref!r} is ambiguous: {names}")
        return matches[0]

    @staticmethod
    def write_run(run_dir: Path, tape: RecordingTape, name: str,
                  meta_extra: dict | None = None) -> Path:
        trace_path = run_dir / "trace.jsonl"
        with trace_path.open("w", encoding="utf-8") as fh:
            for event in tape.events:
                fh.write(event.to_json() + "\n")

        meta = {
            "schema": SCHEMA_VERSION,
            "name": name,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "host": socket.gethostname(),
            "python": platform.python_version(),
            "platform": platform.platform(terse=True),
            "steps": len(tape.events),
            "warnings": list(tape._warnings),
            "redacted": bool(getattr(tape, "redact", None)),
        }
        if meta_extra:
            meta.update(meta_extra)
        (run_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return trace_path

    @staticmethod
    def read_run(run_dir: Path) -> tuple[list[Event], dict]:
        run_dir = Path(run_dir)
        trace_path = run_dir / "trace.jsonl"
        if not trace_path.exists():
            raise FileNotFoundError(f"{trace_path} does not exist")
        events = []
        with trace_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    events.append(Event.from_json(line))
        meta_path = run_dir / "meta.json"
        meta = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return events, meta
