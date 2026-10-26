#!/usr/bin/env python3
"""A real integration: DeepSeek chat through the stdlib, recorded with
redaction on. 5 lines of integration inside run_agent -- everything else is
the same pattern as demo_agent.py.

    export DEEPSEEK_API_KEY=sk-...
    python examples/deepseek_agent.py "Give me a haiku about databases."
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tapedeck

API = "https://api.deepseek.com/chat/completions"


def chat(messages: list[dict], model="deepseek-chat"):
    """stdlib-only chat completion returning (content, usage)."""
    req = urllib.request.Request(
        API,
        data=json.dumps({"model": model, "messages": messages}).encode(),
        headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}",
                 "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.load(resp)
    return data["choices"][0]["message"]["content"], dict(data["usage"], model=model)


def run_agent(tape: "tapedeck.Tape", user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]
    reply = tape.call("model", "deepseek.chat", chat, messages)  # <-- the whole integration
    return tape.call("tool", "postprocess", lambda: reply.strip())


def main() -> int:
    user_message = sys.argv[1] if len(sys.argv) > 1 else "Say something about deterministic replay."
    redactor = tapedeck.Redactor()  # emails / api keys / phone numbers never hit disk
    with tapedeck.record("deepseek", redact=redactor) as tape:
        reply = run_agent(tape, user_message)
        tapedeck.save(tape)
    print(reply)
    print(f"\n[recorded {len(tape.events)} steps -> {tape._run_dir}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
