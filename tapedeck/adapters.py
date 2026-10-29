"""Auto-interception for provider SDKs: hand ``tape.watch(client)`` a client.

The explicit ``tape.call`` interface stays the core; this module removes the
remaining adoption friction for agents that call a provider SDK directly::

    import tapedeck, openai

    client = openai.OpenAI()
    with tapedeck.record("chat-v3") as tape:
        client = tape.watch(client)                 # REC: traced live
        resp = client.chat.completions.create(...)  # the code does not change

    with tapedeck.play("chat-v3") as tape:
        client = tape.watch(client)                 # PLAY: served from tape
        resp = client.chat.completions.create(...)  # no network, no API key

Intercepted method paths:

* ``chat.completions.create`` -- the OpenAI Python SDK (openai>=1), which
  also covers every OpenAI-compatible endpoint (DeepSeek, vLLM, Ollama's
  OpenAI route, ...): point ``base_url`` wherever you already do.
* ``responses.create`` -- the OpenAI Responses API.
* ``messages.create`` -- the Anthropic Python SDK (anthropic>=0.34),
  including ``stream=True``.

Recording stores the response as a plain dict (pydantic ``model_dump``) plus
a normalized ``usage`` block, so cost accounting works unchanged. Replay
rebuilds the SDK's own response type from the recorded dict
(``model_validate``); if the installed SDK version no longer matches the
recording, replay degrades to an attribute shim so ``resp.choices[0].
message.content`` still works -- content access is preserved even when
schema validation is not.

Honest boundaries (same rules as the core, extended here):

* Requests must be JSON-able. Messages, tools and sampling params are;
  pydantic classes or binary payloads are not and will record as ``repr``,
  which replays as an input-mismatch drift (by design: that drift is real).
* ``stream=True`` is recorded chunk by chunk after the stream is consumed;
  the trace event lands when the last chunk arrives, not when ``create``
  returns. OpenAI usage is captured when ``stream_options={"include_usage":
  True}``; Anthropic usage is assembled from ``message_start`` /
  ``message_delta`` chunks.
"""
from __future__ import annotations

import importlib
import time
from typing import Any, Callable

from .session import RecordingTape, ReplayingTape, _jsonable

# Method paths that count as one model call each. Anything else on a watched
# client passes through untouched (token counting, file uploads, your code).
_INTERCEPTED = {
    ("chat", "completions", "create"),
    ("responses", "create"),
    ("messages", "create"),
}

# The SDK response type to rebuild at replay time, per method path.
_RESPONSE_CLASSES = {
    ("chat", "completions", "create"): ("openai.types.chat", "ChatCompletion"),
    ("responses", "create"): ("openai.types.responses", "Response"),
    ("messages", "create"): ("anthropic.types", "Message"),
}

# Same, for the per-chunk events of a recorded stream.
_CHUNK_CLASSES = {
    ("chat", "completions", "create"): ("openai.types.chat", "ChatCompletionChunk"),
    ("responses", "create"): ("openai.types.responses", "ResponseStreamEvent"),
    ("messages", "create"): ("anthropic.types", "MessageStreamEvent"),
}


def watch_client(tape: Any, client: Any) -> Any:
    """Return a transparent proxy over ``client`` wired to ``tape``."""
    return _Proxy(client, tape)


class _Proxy:
    """Transparent attribute proxy; intercepts only the create() paths."""

    def __init__(self, target: Any, tape: Any, path: tuple = ()):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_tape", tape)
        object.__setattr__(self, "_path", path)

    def __getattr__(self, name: str) -> Any:
        target, tape, path = self._target, self._tape, self._path
        value = getattr(target, name)
        nxt = path + (name,)
        if nxt in _INTERCEPTED:
            return _interceptor(tape, nxt, value)
        if callable(value) or isinstance(value, (str, bytes, int, float, bool, list, dict)) or value is None:
            return value
        return _Proxy(value, tape, nxt)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._target, name, value)


def _interceptor(tape: Any, path: tuple, method: Callable) -> Callable:
    if isinstance(tape, ReplayingTape):
        def replay(*args: Any, **kwargs: Any) -> Any:
            request = {"args": _jsonable(list(args)), "kwargs": _jsonable(kwargs)}
            def live(_request):  # lenient mode falls through here; strict never calls it
                return method(*args, **kwargs)
            recorded = tape.call("model", _call_name(kwargs), live, request)
            if kwargs.get("stream"):
                return _ReplayStream(recorded if isinstance(recorded, list) else [], path)
            return _rebuild(recorded, _RESPONSE_CLASSES[path])
        return replay

    def record(*args: Any, **kwargs: Any) -> Any:
        request = {"args": _jsonable(list(args)), "kwargs": _jsonable(kwargs)}
        model = kwargs.get("model")
        holder: dict = {}
        def invoke(_request):
            holder["raw"] = raw = method(*args, **kwargs)
            return raw
        if kwargs.get("stream"):
            return _RecordingStream(tape, _call_name(kwargs), request,
                                    method(*args, **kwargs), time.perf_counter(), path)
        # tape.call returns the transformed (recordable) value; the agent
        # must keep receiving the SDK's own response object, hence the holder.
        tape.call("model", _call_name(kwargs), invoke, request,
                  transform=lambda raw: _serialize(raw, model))
        return holder["raw"]
    return record


def _call_name(kwargs: dict) -> str:
    model = kwargs.get("model")
    return str(model) if model else "model"


def _serialize(response: Any, model: Any) -> tuple[Any, dict | None]:
    """Response object -> (plain dict, normalized usage). Never raises."""
    dump = None
    for attempt in ("model_dump", "dict", "to_dict"):
        fn = getattr(response, attempt, None)
        if not callable(fn):
            continue
        try:
            dump = fn(mode="json") if attempt == "model_dump" else fn()
        except TypeError:
            try:
                dump = fn()
            except Exception:  # noqa: BLE001 - tracing never breaks a live call
                dump = None
        except Exception:  # noqa: BLE001
            dump = None
        if dump is not None:
            break
    if dump is None:
        return _jsonable(response), _usage_of(getattr(response, "usage", None), model)
    return dump, _usage_of(dump.get("usage") if isinstance(dump, dict) else None, model)


def _usage_of(usage: Any, model: Any) -> dict | None:
    """Normalize provider usage into the trace convention (prompt/completion)."""
    if usage is None:
        return None
    if not isinstance(usage, dict):
        try:
            usage = dict(usage)
        except (TypeError, ValueError):
            return None
    normalized = dict(usage)
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    if prompt is not None:
        normalized["prompt_tokens"] = prompt
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    if completion is not None:
        normalized["completion_tokens"] = completion
    if model is not None:
        normalized["model"] = model
    return normalized


def _rebuild(recorded: Any, class_ref: tuple[str, str]) -> Any:
    """Recorded dict -> the SDK's response type; shim if that fails.

    A non-dict ``recorded`` means replay fell through to live (lenient mode)
    and ``tape.call`` handed back the raw SDK object already -- pass it on.
    """
    if not isinstance(recorded, dict):
        return recorded
    try:
        module_name, class_name = class_ref
        cls = getattr(importlib.import_module(module_name), class_name)
        return cls.model_validate(recorded)
    except Exception:  # noqa: BLE001 - SDK version drift must not kill replay
        return _Shim(recorded)


class _Shim:
    """Attribute access over a recorded dict/list tree.

    The fallback when ``model_validate`` rejects a recording (SDK version
    drift): response shape access -- ``resp.choices[0].message.content`` --
    keeps working; ``isinstance`` checks and provider-side helpers will not.
    """

    def __init__(self, value: Any):
        self._value = value

    def __getattr__(self, name: str) -> Any:
        return _wrap(self._value.get(name) if isinstance(self._value, dict) else None)

    def __getitem__(self, key: Any) -> Any:
        return _wrap(self._value[key])

    def __iter__(self):
        return iter(self._value)

    def __len__(self) -> int:
        return len(self._value)

    def __repr__(self) -> str:
        return f"tapedeck.Shim({self._value!r})"

    def model_dump(self, *args: Any, **kwargs: Any) -> Any:
        return self._value


def _wrap(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return _Shim(value)
    return value


class _RecordingStream:
    """Yield chunks live while collecting them for one trace event.

    The event is appended when the stream is consumed -- a replay of a
    streaming call must serve the chunks, and only consuming the live stream
    produces them. If the agent abandons the stream, ``finally`` still writes
    whatever arrived (the flight-recorder contract: crashes are data).
    """

    def __init__(self, tape: RecordingTape, name: str, request: dict,
                 stream: Any, started: float, path: tuple):
        self._tape, self._name, self._request = tape, name, request
        self._stream, self._started, self._path = stream, started, path

    def __iter__(self):
        chunks: list = []
        usage: dict | None = None
        try:
            for chunk in self._stream:
                dump = _chunk_dict(chunk)
                chunks.append(_jsonable(dump))
                found = _stream_usage(dump, self._path)
                if found:
                    # Usage can arrive across several chunks (Anthropic
                    # splits input/output between message_start/delta): merge.
                    usage = {**(usage or {}), **found}
                yield chunk
        finally:
            # Input mirrors tape.call's convention ([request]) so a streaming
            # record matches its own replay byte for byte.
            self._tape.record_raw(
                "model", self._name, [self._request], output=chunks, usage=usage,
                wall_ms=round((time.perf_counter() - self._started) * 1000, 3))

    def __enter__(self):
        enter = getattr(self._stream, "__enter__", None)
        if enter is not None:
            enter()
        return self

    def __exit__(self, *exc):
        exit_ = getattr(self._stream, "__exit__", None)
        return exit_(*exc) if exit_ is not None else False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def _chunk_dict(chunk: Any) -> Any:
    for name in ("model_dump", "dict", "to_dict"):
        fn = getattr(chunk, name, None)
        if callable(fn):
            try:
                return fn(mode="json") if name == "model_dump" else fn()
            except Exception:  # noqa: BLE001
                continue
    return _jsonable(chunk)


def _stream_usage(chunk: dict, path: tuple) -> dict | None:
    """Pull a usage block out of one stream chunk, if it carries one."""
    usage = chunk.get("usage")
    if path == ("messages", "create"):
        if chunk.get("type") == "message_start":
            usage = (chunk.get("message") or {}).get("usage")
        elif chunk.get("type") == "message_delta":
            usage = chunk.get("usage")
    if not usage:
        return None
    return _usage_of(usage, chunk.get("model"))


class _ReplayStream:
    """Re-serve recorded chunks without any network."""

    def __init__(self, chunks: list, path: tuple):
        self._chunks, self._path = chunks, path

    def __iter__(self):
        for chunk in self._chunks:
            yield _rebuild(chunk, _CHUNK_CLASSES[self._path])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
