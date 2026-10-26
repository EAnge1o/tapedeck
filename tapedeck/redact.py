"""Opt-in redaction applied at record time.

Trust rule of the product: **traces never leave your machine**, but traces
are also the thing you paste into an issue or share with a teammate. A
recorded prompt with an API key in it defeats that. So the redactor runs on
every recorded string before it touches disk -- fail-safe: if a pattern
itself misbehaves, the raw value is kept and a warning is surfaced in
meta.json rather than crashing the run.
"""
from __future__ import annotations

import re
from typing import Any

# (label, pattern) -- replaced as <EMAIL>, <PHONE>, ...
DEFAULT_PATTERNS: list[tuple[str, str]] = [
    ("EMAIL", r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    ("API_KEY", r"\b(?:sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    ("CN_PHONE", r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    ("CN_ID", r"(?<!\d)\d{17}[\dXx](?!\d)"),
    ("CARD", r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
]


class Redactor:
    def __init__(self, extra_patterns: list[tuple[str, str]] | None = None,
                 use_defaults: bool = True):
        compiled: list[tuple[str, re.Pattern]] = []
        patterns = (DEFAULT_PATTERNS if use_defaults else []) + (extra_patterns or [])
        for label, pattern in patterns:
            try:
                compiled.append((label, re.compile(pattern)))
            except re.error:
                continue  # a bad custom pattern must not break recording
        self.compiled = compiled

    def scrub_text(self, text: str) -> str:
        for label, pattern in self.compiled:
            text = pattern.sub(f"<{label}>", text)
        return text

    def apply(self, value: Any) -> Any:
        """Recursively scrub every string in JSON-like structures."""
        if isinstance(value, str):
            return self.scrub_text(value)
        if isinstance(value, dict):
            return {k: self.apply(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.apply(v) for v in value]
        return value
