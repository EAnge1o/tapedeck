"""Diff two recorded runs -- the reason the tool exists.

Alignment is positional (step i vs step i), which mirrors how an agent
actually drifts: everything matches until some step takes a different turn,
after which comparing by content would be noise. Output drift past the
first divergence is *summarized*, not dumped, so the report answers the one
question that matters: **where did it go wrong first?**
"""
from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .costs import total_cost
from .session import Event
from .store import RunStore

EXIT_SAME = 0
EXIT_OUTPUT_DRIFT = 1
EXIT_STRUCTURE_DRIFT = 2


@dataclass
class DiffResult:
    same: bool = True
    structural: bool = False
    first_divergence: int | None = None
    reason: str | None = None
    lines: list[str] = field(default_factory=list)


def _fmt(value, limit: int = 240) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def diff_runs(a_dir, b_dir, context: int = 2, prices: dict | None = None) -> DiffResult:
    events_a, meta_a = RunStore.read_run(Path(a_dir))
    events_b, meta_b = RunStore.read_run(Path(b_dir))
    prices = prices or {}
    result = DiffResult()

    cost_a = total_cost(events_a, prices)
    cost_b = total_cost(events_b, prices)

    result.lines.append(f"run A: {meta_a.get('name', a_dir)} ({len(events_a)} steps)")
    result.lines.append(f"run B: {meta_b.get('name', b_dir)} ({len(events_b)} steps)")

    n = max(len(events_a), len(events_b))
    for i in range(n):
        ea = events_a[i] if i < len(events_a) else None
        eb = events_b[i] if i < len(events_b) else None
        if ea is None or eb is None:
            result.same = False
            result.structural = True
            result.first_divergence = result.first_divergence if result.first_divergence is not None else i
            result.reason = "runs have different step counts"
            present, absent = ("B", "A") if eb else ("A", "B")
            event = eb or ea
            result.lines.append(f"step {i}: exists only in {present} -- {event.kind}:{event.name}")
            break
        if ea.kind != eb.kind or ea.name != eb.name:
            result.same = False
            result.structural = True
            result.first_divergence = i
            result.reason = "steps take different call sites"
            result.lines.append(f"step {i}: A={ea.kind}:{ea.name}  B={eb.kind}:{eb.name}")
            break
        usage_delta = _usage_delta(ea.usage, eb.usage)
        if ea.error or eb.error:
            if ea.error != eb.error:
                result.same = False
                result.first_divergence = result.first_divergence if result.first_divergence is not None else i
                result.reason = result.reason or "an error appeared or disappeared"
                result.lines.append(f"step {i} ({ea.kind}:{ea.name}) error:")
                result.lines.append(f"  A: {ea.error}")
                result.lines.append(f"  B: {eb.error}")
                result.lines.extend(_context_lines(events_a, events_b, i, context))
                break
            continue  # both failed identically: keep scanning
        if not _outputs_equal(ea.output, eb.output):
            result.same = False
            result.first_divergence = i
            result.reason = f"outputs diverge at {ea.kind}:{ea.name}"
            result.lines.append(f"step {i} ({ea.kind}:{ea.name}) output:")
            result.lines.append(f"  A: {_fmt(ea.output)}")
            result.lines.append(f"  B: {_fmt(eb.output)}")
            result.lines.extend(_context_lines(events_a, events_b, i, context))
            if usage_delta:
                result.lines.append(f"  usage: {usage_delta}")
            break

    result.lines.append(_cost_line("A", cost_a))
    result.lines.append(_cost_line("B", cost_b))
    if result.same:
        result.lines.append("runs are identical")
    return result


def _outputs_equal(a, b) -> bool:
    try:
        return json.loads(json.dumps(a)) == json.loads(json.dumps(b))
    except (TypeError, ValueError):
        return a == b


def _usage_delta(ua, ub) -> str | None:
    if not ua and not ub:
        return None
    keys = set((ua or {}).keys()) | set((ub or {}).keys())
    keys -= {"model"}
    parts = [f"{k}: {((ua or {}).get(k) or 0)} -> {((ub or {}).get(k) or 0)}"
             for k in sorted(keys)
             if (ua or {}).get(k) != (ub or {}).get(k)]
    return "; ".join(parts) if parts else None


def _cost_line(tag: str, cost: dict) -> str:
    total = cost.get("total_usd")
    if total is None:
        tokens = sum(m["prompt_tokens"] + m["completion_tokens"] for m in cost["per_model"].values())
        return f"run {tag}: {tokens} tokens (no price table for these models)"
    return f"run {tag}: ${total:.4f}"


def _context_lines(events_a: list[Event], events_b: list[Event], at: int, context: int) -> list[str]:
    """Show the inputs that led to the divergence -- usually the actual culprit."""
    lo = max(0, at - context)
    lines = [f"  context (steps {lo}..{at - 1}):"]
    for i in range(lo, at):
        event = events_a[i]
        lines.append(f"    step {i} ({event.kind}:{event.name}) input: {_fmt(event.input, 120)}")
    return lines
