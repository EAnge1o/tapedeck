"""The record / replay core.

Two tape implementations share one interface, so an agent can be written
once against ``Tape.call`` and run either live or offline:

* :class:`RecordingTape` -- wraps a live run, appending one JSONL event per
  :meth:`~RecordingTape.call`.
* :class:`ReplayingTape` -- serves recorded outputs back without ever
  invoking the wrapped callable, so reruns cost nothing and are
  deterministic. Drift (a changed input, a new or vanished step) is raised
  as :class:`ReplayDrift` with the exact step that changed.

Tracing must never break a run: every recording failure is captured as a
warning event and the original result is still returned.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

SCHEMA_VERSION = 1


class ReplayDrift(RuntimeError):
    """A replayed run diverged from its recording.

    Attributes:
        step:     index of the offending event.
        kind/name: the call site.
        reason:   "missing" | "input_mismatch" | "kind_order".
        detail:   human-readable explanation of what changed.
    """

    def __init__(self, step: int, kind: str, name: str, reason: str, detail: str):
        self.step, self.kind, self.name, self.reason, self.detail = step, kind, name, reason, detail
        super().__init__(
            f"replay drift at step {step} ({kind}:{name}): {reason} -- {detail}"
        )


def _jsonable(value: Any) -> Any:
    """Best-effort JSON encoding of arbitrary inputs/outputs.

    Anything a provider or tool returns should pass through unchanged; the
    fallback (repr string) exists so tracing never crashes a live run.
    """
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


@dataclass
class Event:
    """One recorded step: a model call, a tool call, or a marker."""

    i: int
    kind: str                    # "model" | "tool" | "mark"
    name: str
    input: Any = None
    output: Any = None
    error: str | None = None
    usage: dict | None = None    # {"prompt_tokens": n, "completion_tokens": n, ...}
    wall_ms: float = 0.0

    def to_json(self) -> str:
        return json.dumps(
            {
                "v": SCHEMA_VERSION,
                "i": self.i,
                "kind": self.kind,
                "name": self.name,
                "input": _jsonable(self.input),
                "output": _jsonable(self.output),
                "error": self.error,
                "usage": self.usage,
                "wall_ms": round(self.wall_ms, 3),
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, line: str) -> "Event":
        data = json.loads(line)
        if data.get("v") != SCHEMA_VERSION:
            raise ValueError(f"unsupported trace schema version: {data.get('v')!r}")
        return cls(
            i=data["i"], kind=data["kind"], name=data["name"], input=data.get("input"),
            output=data.get("output"), error=data.get("error"),
            usage=data.get("usage"), wall_ms=data.get("wall_ms", 0.0),
        )


class Tape:
    """The interface agents code against: ``result = tape.call(kind, name, fn, *args)``."""

    def call(self, kind: str, name: str, fn: Callable[..., Any], *args,
             transform: Callable[[Any], tuple[Any, dict | None]] | None = None,
             **kwargs) -> Any:
        raise NotImplementedError

    def watch(self, client: Any) -> Any:
        """Wire a provider SDK client to this tape (OpenAI/Anthropic/compatible).

        Returns a transparent proxy: the code keeps reading like plain SDK
        calls, while every intercepted completion is traced under kind
        ``model``. On a recording tape calls run live; on a replaying tape
        they are served from the trace without touching the network.
        Requires the matching optional extra (``pip install
        tapedeck-llm[openai]`` / ``[anthropic]``) -- the SDK itself is
        imported only here, never by the core.
        """
        from .adapters import watch_client
        return watch_client(self, client)

    def mark(self, name: str) -> None:
        """Record a zero-cost landmark (phase boundaries, retries, decisions)."""
        self.call("mark", name, lambda: None)


@dataclass
class RecordingTape(Tape):
    """Append one event per call while executing everything live."""

    redact: Any = None            # optional Redactor
    _events: list = field(default_factory=list)
    _warnings: list = field(default_factory=list)

    def call(self, kind: str, name: str, fn: Callable[..., Any], *args,
             transform: Callable[[Any], tuple[Any, dict | None]] | None = None,
             **kwargs) -> Any:
        i = len(self._events)
        started = time.perf_counter()
        try:
            output = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - failures are data too
            wall = (time.perf_counter() - started) * 1000
            self._events.append(Event(
                i=i, kind=kind, name=name,
                input=self._redact_in(args, kwargs),
                error=f"{type(exc).__name__}: {exc}",
                usage=getattr(exc, "usage", None),
                wall_ms=wall,
            ))
            raise
        wall = (time.perf_counter() - started) * 1000
        usage = None
        result = output
        # Providers return (payload, usage) pairs; accept both conventions.
        if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], dict):
            result, usage = output
        if transform is not None:
            # SDK adapters: raw response object -> (recordable dict, usage).
            result, usage = transform(result)
        self._events.append(Event(
            i=i, kind=kind, name=name,
            input=self._redact_in(args, kwargs),
            output=self._redact_out(result),
            usage=usage,
            wall_ms=wall,
        ))
        return result

    def record_raw(self, kind: str, name: str, input: Any, output: Any = None,
                   error: str | None = None, usage: dict | None = None,
                   wall_ms: float = 0.0) -> None:
        """Append a pre-computed event.

        The one escape hatch from ``call`` -- used by the SDK adapters for
        streaming responses, where the trace event can only be written after
        the stream has been consumed.
        """
        redact = self.redact.apply if self.redact else _jsonable
        self._events.append(Event(
            i=len(self._events), kind=kind, name=name,
            input=redact(input),
            output=redact(output) if output is not None else None,
            error=error, usage=usage, wall_ms=wall_ms,
        ))

    def _redact_in(self, args: tuple, kwargs: dict) -> Any:
        payload: Any = {"args": list(args), "kwargs": kwargs} if kwargs else list(args)
        return self.redact.apply(payload) if self.redact else _jsonable(payload)

    def _redact_out(self, value: Any) -> Any:
        return self.redact.apply(value) if self.redact else _jsonable(value)

    @property
    def events(self) -> list[Event]:
        return list(self._events)

    def __enter__(self) -> "RecordingTape":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Auto-save on clean exit; on failure the partial trace is still
        # saved (with the error events in it) -- that is the flight-recorder
        # contract: crashes are exactly what you want on tape.
        try:
            import tapedeck
            tapedeck.save(self)
        except Exception:  # noqa: BLE001 - saving must never mask the real error
            pass
        return False


@dataclass
class ReplayingTape(Tape):
    """Serve recorded outputs; never call ``fn`` unless ``lenient`` allows it.

    ``strict=True`` (default) turns any divergence -- changed input, missing
    step, extra step -- into :class:`ReplayDrift`, which is what makes replay
    usable as a regression test. ``lenient=True`` executes the real callable
    at the first divergence-free opportunity and records nothing; it exists
    for hybrid debugging, not for CI.
    """

    events: list = field(default_factory=list)
    strict: bool = True
    lenient: bool = False
    cursor: int = 0
    served: int = 0
    drifts: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.lenient:
            self.strict = False

    def call(self, kind: str, name: str, fn: Callable[..., Any], *args,
             transform: Callable[[Any], tuple[Any, dict | None]] | None = None,
             **kwargs) -> Any:
        if self.cursor >= len(self.events):
            return self._drift(len(self.events), kind, name, "missing",
                               "the recording ends before this call",
                               fn=fn, args=args, kwargs=kwargs)
        event = self.events[self.cursor]
        if event.kind != kind or event.name != name:
            return self._drift(event.i, kind, name, "kind_order",
                               f"recording has {event.kind}:{event.name} at this position",
                               fn=fn, args=args, kwargs=kwargs)
        want = event.input
        got = args if not kwargs else {"args": list(args), "kwargs": kwargs}
        if not _deep_equal(want, got):
            return self._drift(event.i, kind, name, "input_mismatch",
                               f"recorded input {want!r} != live input {got!r}",
                               fn=fn, args=args, kwargs=kwargs)
        self.cursor += 1
        self.served += 1
        if event.error:
            raise ReplayDrift(event.i, kind, name, "recorded_error",
                              f"the original run failed here: {event.error}")
        return event.output

    def _drift(self, step: int, kind: str, name: str, reason: str, detail: str,
               fn: Callable[..., Any] | None = None, args: tuple = (), kwargs: dict | None = None):
        """Handle a divergence: raise (strict) or fall through to live (lenient)."""
        drift = ReplayDrift(step, kind, name, reason, detail)
        self.drifts.append(drift)
        if self.strict:
            raise drift
        return fn(*args, **(kwargs or {})) if fn is not None else None

    def assert_complete(self) -> None:
        """Fail if the recording had steps the replay never reached."""
        if self.cursor < len(self.events):
            event = self.events[self.cursor]
            raise ReplayDrift(
                event.i, event.kind, event.name, "unreached",
                f"{len(self.events) - self.cursor} recorded step(s) were never replayed, "
                f"first: {event.kind}:{event.name}",
            )

    def __enter__(self) -> "ReplayingTape":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.assert_complete()
        return False


def _deep_equal(a: Any, b: Any) -> bool:
    """JSON-round-trip equality: 1 == 1.0, tuples == lists, {"a": 1} == {"a": 1}."""
    try:
        return json.loads(json.dumps(a)) == json.loads(json.dumps(b))
    except (TypeError, ValueError):
        return a == b
