#!/usr/bin/env python3
"""A tiny deterministic agent that shows the whole product in one file.

The "model" is scripted (no network, no API key): a plan step, two tool
calls, an answer step -- each traced through the tape. Run it three ways:

    python examples/demo_agent.py --mode record --scenario stable   # run A
    python examples/demo_agent.py --mode record --scenario drift    # run B (policy changed)
    python examples/demo_agent.py --mode replay                     # rerun latest offline, $0.00

    python -m tapedeck diff <run-A> <run-B>        # where did it drift, first?
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tapedeck

CATALOG = {"coffee": 3.50, "sandwich": 8.90, "cookie": 1.20}
# What a provider would report back: usage is kept verbatim in the trace and
# priced by tapedeck's table (see `python -m tapedeck ls`).
MODEL_USAGE = {"prompt_tokens": 118, "completion_tokens": 24, "model": "deepseek-chat"}


def scripted_model(tape, step: str, payload):
    """A deterministic stand-in for a real model call (returns payload + usage)."""
    def call():
        if step == "plan":
            return {"tool": "get_price", "args": payload["item"], "thought": "need unit price"}
        if step == "answer":
            discount = payload["discount"]
            text = (f"Your {payload['item']} is ${payload['price']:.2f}; "
                    f"with the {round((1 - discount) * 100)}% member discount you pay "
                    f"${payload['total']:.2f}.")
            return text
        raise ValueError(step)
    return tape.call("model", f"model.{step}", lambda: (call(), dict(MODEL_USAGE)))


def get_price(tape, item: str):
    return tape.call("tool", "get_price", lambda: CATALOG[item])


def compute_total(tape, price: float, discount: float, qty: int):
    return tape.call("tool", "compute_total", lambda: round(price * qty * discount, 2))


def run_agent(tape: "tapedeck.Tape", item: str, qty: int, discount: float) -> str:
    plan = scripted_model(tape, "plan", {"item": item})
    assert plan["tool"] == "get_price"
    price = get_price(tape, plan["args"])
    total = compute_total(tape, price, discount, qty)
    answer = scripted_model(tape, "answer",
                            {"item": item, "price": price, "total": total, "discount": discount})
    return answer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["record", "replay"], default="record")
    parser.add_argument("--scenario", choices=["stable", "drift"], default="stable")
    parser.add_argument("--item", default="coffee")
    parser.add_argument("--qty", type=int, default=2)
    args = parser.parse_args()

    if args.mode == "record":
        # "drift" = someone changed the member-discount policy from 10% to 20%.
        discount = 0.80 if args.scenario == "drift" else 0.90
        with tapedeck.record(f"demo-{args.scenario}") as tape:
            text = run_agent(tape, args.item, args.qty, discount)
            tapedeck.save(tape)
        tokens = sum(e.usage.get("prompt_tokens", 0) + e.usage.get("completion_tokens", 0)
                     for e in tape.events if e.usage)
        print(text)
        print(f"\n[recorded {len(tape.events)} steps -> {tape._run_dir} | {tokens} tokens]")
        return 0

    # replay: strict, offline, free. Replays the latest recorded run; the
    # scenario comes from the run name so the replay matches its recording.
    runs = tapedeck.RunStore().list_runs()
    if not runs:
        print("nothing recorded yet -- run with --mode record first", file=sys.stderr)
        return 2
    ref = runs[-1].name
    discount = 0.80 if "drift" in ref else 0.90
    tape = tapedeck.play(ref)
    text = run_agent(tape, args.item, args.qty, discount)
    tape.assert_complete()
    print(text)
    print(f"\n[replayed '{ref}': served {tape.served} steps, 0 live calls, $0.00 spent]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
