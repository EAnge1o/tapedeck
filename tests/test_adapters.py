"""Adapter tests: tape.watch() record/replay identity without real SDKs.

The suite injects minimal stand-in modules for ``openai`` / ``anthropic``
into sys.modules, mimicking the one surface the adapters touch:
``model_dump`` on responses/chunks and ``model_validate`` on response
classes. No network, no pydantic, no SDK install -- the dependency-free
test policy holds. Run:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tapedeck
from tapedeck.redact import Redactor
from tapedeck.session import ReplayDrift

import json
import shutil
import tempfile

ROOT = Path(tempfile.mkdtemp(prefix="tapedeck-adapter-tests-"))


class Base(unittest.TestCase):
    """Same per-test sandbox as the core suite (kept local: discovery imports
    both files as top-level modules, so cross-importing is fragile)."""

    def setUp(self):
        self.root = ROOT / self.id().split(".")[-1] / self._testMethodName
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)


# --------------------------------------------------------------------------
# Fake SDK surface
# --------------------------------------------------------------------------
class _PydLike:
    """Stands in for a pydantic v2 model on the one surface tapedeck uses."""

    def __init__(self, data: dict):
        self._data = data
        for key, value in data.items():
            setattr(self, key, _unshallow(value))

    def model_dump(self, mode=None, **_):
        return self._data

    @classmethod
    def model_validate(cls, data: dict):
        return cls(data)

    def __eq__(self, other):
        return isinstance(other, _PydLike) and self._data == other._data


def _unshallow(value):
    if isinstance(value, dict):
        return _PydLike(value)
    if isinstance(value, list):
        return [_unshallow(v) for v in value]
    return value


def _install_fake_sdk(monkey: dict, *, validate_fails: bool = False):
    """Register fake openai/anthropic modules; return an undo callable."""
    class ChatCompletion(_PydLike):
        @classmethod
        def model_validate(cls, data):
            if validate_fails:
                raise ValueError("schema drift (test-simulated)")
            return super().model_validate(data)

    class ChatCompletionChunk(_PydLike):
        pass

    def openai_create(model="gpt-fake", messages=None, stream=False, **kwargs):
        monkey["live_calls"] += 1
        if stream:
            return [_chunk(0, "Hel"), _chunk(1, "lo"), _usage_chunk()]
        return ChatCompletion({
            "id": "chatcmpl-1", "object": "chat.completion", "created": 1700000000,
            "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "Hello!"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3},
        })

    class Completions:
        create = staticmethod(openai_create)
    class Chat:
        completions = Completions()

    class Message(_PydLike):
        pass

    def anthropic_create(model="claude-fake", messages=None, stream=False, **kwargs):
        monkey["live_calls"] += 1
        if stream:
            return [
                _PydLike({"type": "message_start",
                          "message": {"role": "assistant", "usage": {"input_tokens": 9}}}),
                _PydLike({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}}),
                _PydLike({"type": "message_delta", "usage": {"output_tokens": 2}}),
            ]
        return Message({
            "id": "msg_1", "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": "Hello!"}],
            "usage": {"input_tokens": 9, "output_tokens": 2},
        })

    class AnthropicMessages:
        create = staticmethod(anthropic_create)
    class AnthropicClient:
        messages = AnthropicMessages()

    class Embeddings:
        @staticmethod
        def create(model="e", input="x"):
            monkey["live_calls"] += 1
            return _PydLike({"object": "embedding", "model": model, "embedding": [0.0]})

    class OpenAIClient2:
        chat = Chat()
        embeddings = Embeddings()

    openai = types.ModuleType("openai")
    openai_types = types.ModuleType("openai.types")
    openai_chat = types.ModuleType("openai.types.chat")
    openai_chat.ChatCompletion = ChatCompletion
    openai_chat.ChatCompletionChunk = ChatCompletionChunk
    openai.OpenAI = lambda **kw: OpenAIClient2()

    anthropic = types.ModuleType("anthropic")
    anthropic_types = types.ModuleType("anthropic.types")
    anthropic_types.Message = Message
    anthropic.Anthropic = lambda **kw: AnthropicClient()

    saved = {name: sys.modules.get(name) for name in
             ("openai", "openai.types", "openai.types.chat",
              "anthropic", "anthropic.types")}
    sys.modules.update({
        "openai": openai, "openai.types": openai_types, "openai.types.chat": openai_chat,
        "anthropic": anthropic, "anthropic.types": anthropic_types,
    })

    def undo():
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return undo


def _chunk(i, text):
    return _PydLike({"id": "chatcmpl-1", "object": "chat.completion.chunk",
                     "index": i, "choices": [{"index": 0, "delta": {"content": text}}]})


def _usage_chunk():
    return _PydLike({"id": "chatcmpl-1", "object": "chat.completion.chunk",
                     "index": 2, "choices": [],
                     "usage": {"prompt_tokens": 11, "completion_tokens": 3}})


# --------------------------------------------------------------------------
class TestWatchOpenAI(Base):
    def setUp(self):
        super().setUp()
        self.monkey = {"live_calls": 0}
        self._undo = _install_fake_sdk(self.monkey)

    def tearDown(self):
        self._undo()
        super().tearDown()

    def _agent(self, client, question):
        return client.chat.completions.create(
            model="gpt-fake", messages=[{"role": "user", "content": question}])

    def test_record_traces_a_live_sdk_call(self):
        with tapedeck.record("run", root=self.root) as tape:
            client = tape.watch(self._make_client())
            resp = self._agent(client, "hi")
            tapedeck.save(tape)
        self.assertEqual(self.monkey["live_calls"], 1)
        self.assertEqual(resp.choices[0].message.content, "Hello!")
        events = tapedeck.load_run("run", root=self.root)[0]
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual((event.kind, event.name), ("model", "gpt-fake"))
        self.assertEqual(event.input,
                         [{"args": [], "kwargs": {"model": "gpt-fake",
                          "messages": [{"role": "user", "content": "hi"}]}}])
        self.assertEqual(event.output["choices"][0]["message"]["content"], "Hello!")
        self.assertEqual(event.usage["prompt_tokens"], 11)
        self.assertEqual(event.usage["model"], "gpt-fake")

    def _make_client(self):
        import openai
        return openai.OpenAI()

    def test_replay_serves_recorded_response_without_network(self):
        with tapedeck.record("run", root=self.root) as tape:
            self._agent(tape.watch(self._make_client()), "hi")
            tapedeck.save(tape)

        with tapedeck.play("run", root=self.root) as tape2:
            client = tape2.watch(self._make_client())
            resp = self._agent(client, "hi")   # same request -> replayed
        self.assertEqual(self.monkey["live_calls"], 1)  # only the recording ran live
        self.assertEqual(resp.choices[0].message.content, "Hello!")
        self.assertEqual(resp.usage.prompt_tokens, 11)

    def test_replay_flags_changed_request_as_drift(self):
        with tapedeck.record("run", root=self.root) as tape:
            self._agent(tape.watch(self._make_client()), "hi")
            tapedeck.save(tape)
        tape2 = tapedeck.play("run", root=self.root)
        client = tape2.watch(self._make_client())
        with self.assertRaises(ReplayDrift):
            self._agent(client, "a different question")

    def test_lenient_replay_falls_through_to_live(self):
        with tapedeck.record("run", root=self.root) as tape:
            self._agent(tape.watch(self._make_client()), "hi")
            tapedeck.save(tape)
        tape2 = tapedeck.play("run", root=self.root, lenient=True)
        client = tape2.watch(self._make_client())
        resp = self._agent(client, "a different question")
        self.assertEqual(self.monkey["live_calls"], 2)
        self.assertEqual(resp.choices[0].message.content, "Hello!")

    def test_sdk_failure_is_recorded_and_reraised(self):
        def broken(**kwargs):
            self.monkey["live_calls"] += 1
            raise RuntimeError("api down")
        self._make_client().chat.completions.create = staticmethod(broken)  # type: ignore[attr-defined]
        with tapedeck.record("run", root=self.root) as tape:
            client = tape.watch(self._make_client())
            with self.assertRaises(RuntimeError):
                client.chat.completions.create(model="gpt-fake", messages=[{"role": "user"}])
            tapedeck.save(tape)
        events = tapedeck.load_run("run", root=self.root)[0]
        self.assertIn("RuntimeError: api down", events[0].error)

    def test_streaming_records_chunks_and_usage(self):
        with tapedeck.record("run", root=self.root) as tape:
            client = tape.watch(self._make_client())
            stream = client.chat.completions.create(
                model="gpt-fake", messages=[{"role": "user", "content": "hi"}], stream=True)
            text = "".join(c.choices[0].delta.content for c in stream if c.choices)
            tapedeck.save(tape)
        self.assertEqual(text, "Hello")
        events = tapedeck.load_run("run", root=self.root)[0]
        self.assertEqual(events[0].kind, "model")
        self.assertEqual(len(events[0].output), 3)           # three chunks on tape
        self.assertEqual(events[0].usage["completion_tokens"], 3)

    def test_streaming_replay_serves_recorded_chunks(self):
        with tapedeck.record("run", root=self.root) as tape:
            client = tape.watch(self._make_client())
            stream = client.chat.completions.create(
                model="gpt-fake", messages=[{"role": "user", "content": "hi"}], stream=True)
            "".join(c.choices[0].delta.content for c in stream if c.choices)
            tapedeck.save(tape)

        with tapedeck.play("run", root=self.root) as tape2:
            client = tape2.watch(self._make_client())
            stream = client.chat.completions.create(
                model="gpt-fake", messages=[{"role": "user", "content": "hi"}], stream=True)
            text = "".join(c.choices[0].delta.content for c in stream if c.choices)
        self.assertEqual(self.monkey["live_calls"], 1)
        self.assertEqual(text, "Hello")

    def test_unintercepted_paths_pass_through(self):
        with tapedeck.record("run", root=self.root) as tape:
            client = tape.watch(self._make_client())
            client.embeddings.create(model="e", input="x")   # not intercepted
            client.api_key = "sk-xyz"                        # attr passthrough
            tapedeck.save(tape)
        self.assertEqual(self.monkey["live_calls"], 1)       # ...and it ran live
        events = tapedeck.load_run("run", root=self.root)[0]
        self.assertEqual(len(events), 0)

    def test_redaction_applies_to_recorded_requests(self):
        with tapedeck.record("run", root=self.root, redact=Redactor()) as tape:
            self._agent(tape.watch(self._make_client()),
                        "my key is sk-abcdefghijklmnopqrst")
            tapedeck.save(tape)
        event = tapedeck.load_run("run", root=self.root)[0][0]
        self.assertIn("<API_KEY>", json_dumps(event.input))

    def test_shim_fallback_when_sdk_schema_drifts(self):
        with tapedeck.record("run", root=self.root) as tape:
            self._agent(tape.watch(self._make_client()), "hi")
            tapedeck.save(tape)
        # Simulate SDK schema drift: the response class refuses to validate.
        self._undo()  # pop the working fakes first...
        self.monkey["live_calls"] = 0
        self._undo = _install_fake_sdk(self.monkey, validate_fails=True)
        with tapedeck.play("run", root=self.root) as tape2:
            client = tape2.watch(self._make_client())
            resp = self._agent(client, "hi")
        self.assertEqual(resp.choices[0].message.content, "Hello!")


def json_dumps(value):
    import json
    return json.dumps(value, ensure_ascii=False)


class TestWatchAnthropic(Base):
    def setUp(self):
        super().setUp()
        self.monkey = {"live_calls": 0}
        self._undo = _install_fake_sdk(self.monkey)

    def tearDown(self):
        self._undo()
        super().tearDown()

    def test_anthropic_usage_is_normalized(self):
        import anthropic
        with tapedeck.record("run", root=self.root) as tape:
            client = tape.watch(anthropic.Anthropic())
            client.messages.create(model="claude-fake",
                                   messages=[{"role": "user", "content": "hi"}])
            tapedeck.save(tape)
        event = tapedeck.load_run("run", root=self.root)[0][0]
        self.assertEqual((event.kind, event.name), ("model", "claude-fake"))
        self.assertEqual(event.usage["prompt_tokens"], 9)
        self.assertEqual(event.usage["completion_tokens"], 2)

    def test_anthropic_streaming_usage_assembled(self):
        import anthropic
        with tapedeck.record("run", root=self.root) as tape:
            client = tape.watch(anthropic.Anthropic())
            stream = client.messages.create(model="claude-fake",
                                            messages=[{"role": "user"}], stream=True)
            "".join(c.delta.text for c in stream if c.type == "content_block_delta")
            tapedeck.save(tape)
        event = tapedeck.load_run("run", root=self.root)[0][0]
        self.assertEqual(event.usage["prompt_tokens"], 9)
        self.assertEqual(event.usage["completion_tokens"], 2)


if __name__ == "__main__":
    unittest.main()
