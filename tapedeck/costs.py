"""Cost accounting from recorded usage blocks.

The engine never prices anything by itself: it reads the ``usage`` dict the
provider returned (kept verbatim in the trace) and multiplies it by a price
table you own. Prices change; the trace stays honest.
"""
from __future__ import annotations

import json
from pathlib import Path

from .session import Event

# Built-in reference prices, USD per 1M tokens (editable, not authoritative).
BUILTIN_PRICES: dict[str, dict[str, float]] = {
    "deepseek-chat": {"prompt": 0.27, "completion": 1.10},
    "deepseek-reasoner": {"prompt": 0.55, "completion": 2.19},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "gpt-5": {"prompt": 1.25, "completion": 10.00},
}


def load_prices(path: Path | None) -> dict[str, dict[str, float]]:
    prices = dict(BUILTIN_PRICES)
    if path and Path(path).exists():
        prices.update(json.loads(Path(path).read_text(encoding="utf-8")))
    return prices


def cost_of(model: str, usage: dict | None, prices: dict[str, dict[str, float]]) -> float | None:
    if not usage:
        return None
    table = prices.get(model)
    if not table:
        return None
    prompt = usage.get("prompt_tokens") or 0
    completion = usage.get("completion_tokens") or 0
    cached = usage.get("prompt_cache_hit_tokens") or 0
    billable_prompt = max(prompt - cached, 0)
    return (
        billable_prompt * table.get("prompt", 0.0)
        + cached * table.get("cached_prompt", table.get("prompt", 0.0))
        + completion * table.get("completion", 0.0)
    ) / 1_000_000


def total_cost(events: list[Event], prices: dict[str, dict[str, float]] | None = None) -> dict:
    """Sum cost/usage across a run.

    ``prices=None`` uses the built-in reference table. Usage blocks are
    stored under event.usage and carry an optional ``model`` key (set by the
    caller when recording model calls); events without usage contribute zero.
    """
    if prices is None:
        prices = BUILTIN_PRICES
    per_model: dict[str, dict[str, float]] = {}
    total = 0.0
    priced = False
    for event in events:
        usage = event.usage
        if not usage:
            continue
        model = usage.get("model", "unknown")
        bucket = per_model.setdefault(
            model, {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0, "usd": 0.0}
        )
        bucket["prompt_tokens"] += usage.get("prompt_tokens") or 0
        bucket["completion_tokens"] += usage.get("completion_tokens") or 0
        bucket["calls"] += 1
        c = cost_of(model, usage, prices)
        if c is not None:
            bucket["usd"] += c
            total += c
            priced = True
    return {"total_usd": round(total, 6) if priced else None, "per_model": per_model}
