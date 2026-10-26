"""``python -m tapedeck``: ls / report / diff / prune.

The CLI operates on recorded runs only -- recording and replay happen in
your agent code (two function calls, see README). Exit codes on ``diff``
are designed for CI: 0 same, 1 output drift, 2 structural drift.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import __version__
from .costs import load_prices, total_cost
from .diff import EXIT_OUTPUT_DRIFT, EXIT_SAME, EXIT_STRUCTURE_DRIFT, diff_runs
from .store import DEFAULT_ROOT, RunStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tapedeck",
        description="Flight recorder for LLM agents: record once, replay forever, diff what drifted.",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help=f"run storage root (default {DEFAULT_ROOT})")
    parser.add_argument("--prices", type=Path, default=None,
                        help="JSON price table {model: {prompt: usd/1M, completion: usd/1M}}")
    parser.add_argument("--version", action="version", version=f"tapedeck {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ls = sub.add_parser("ls", help="list recorded runs")
    p_ls.add_argument("--json", action="store_true", dest="as_json")

    p_report = sub.add_parser("report", help="per-step table for one run")
    p_report.add_argument("run")

    p_diff = sub.add_parser("diff", help="compare two runs; exit 1 on output drift, 2 on structure drift")
    p_diff.add_argument("a")
    p_diff.add_argument("b")
    p_diff.add_argument("--context", type=int, default=2,
                        help="how many leading steps to show before the divergence")
    p_diff.add_argument("--json", action="store_true", dest="as_json")

    p_prune = sub.add_parser("prune", help="delete old runs, keep the newest --keep")
    p_prune.add_argument("--keep", type=int, default=10)

    return parser


def cmd_ls(store: RunStore, prices: dict, as_json: bool) -> int:
    runs = store.list_runs()
    rows = []
    for run_dir in runs:
        try:
            events, meta = RunStore.read_run(run_dir)
        except (ValueError, OSError):
            continue
        cost = total_cost(events, prices)
        rows.append({
            "run": run_dir.name,
            "name": meta.get("name"),
            "steps": len(events),
            "tokens": sum(
                (e.usage.get("prompt_tokens", 0) + e.usage.get("completion_tokens", 0))
                for e in events if e.usage
            ),
            "usd": cost.get("total_usd"),
            "redacted": meta.get("redacted", False),
        })
    if as_json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0
    if not rows:
        print(f"no runs under {store.root}")
        return 0
    width = max(len(r["run"]) for r in rows) + 2
    print(f"{'run':<{width}} {'steps':>5} {'tokens':>9} {'usd':>10}  name")
    for r in rows:
        usd = f"{r['usd']:.4f}" if r["usd"] is not None else "-"
        flag = " [redacted]" if r["redacted"] else ""
        print(f"{r['run']:<{width}} {r['steps']:>5} {r['tokens']:>9} {usd:>10}  {r['name'] or ''}{flag}")
    return 0


def cmd_report(store: RunStore, prices: dict, run_ref: str) -> int:
    run_dir = store.resolve(run_ref)
    events, meta = RunStore.read_run(run_dir)
    cost = total_cost(events, prices)
    print(f"run {run_dir.name}")
    print(f"  recorded: {meta.get('recorded_at', '?')}  host: {meta.get('host', '?')}  "
          f"python: {meta.get('python', '?')}")
    if meta.get("redacted"):
        print("  [redacted at record time]")
    print(f"  {'#':>3}  {'kind':<6} {'name':<28} {'wall_ms':>8}  usage")
    for e in events:
        usage = ""
        if e.usage:
            usage = f"p={e.usage.get('prompt_tokens', 0)} c={e.usage.get('completion_tokens', 0)}"
        print(f"  {e.i:>3}  {e.kind:<6} {e.name:<28} {e.wall_ms:>8.1f}  {usage}"
              + (f"  ERROR: {e.error}" if e.error else ""))
    total = cost.get("total_usd")
    if total is not None:
        print(f"  total cost: ${total:.4f}")
    return 0


def cmd_diff(store: RunStore, prices: dict, a: str, b: str, context: int, as_json: bool) -> int:
    a_dir, b_dir = store.resolve(a), store.resolve(b)
    result = diff_runs(a_dir, b_dir, context=context, prices=prices)
    if as_json:
        print(json.dumps({
            "same": result.same,
            "structural": result.structural,
            "first_divergence": result.first_divergence,
            "reason": result.reason,
        }, ensure_ascii=False, indent=2))
    else:
        print("\n".join(result.lines))
    if result.same:
        return EXIT_SAME
    return EXIT_STRUCTURE_DRIFT if result.structural else EXIT_OUTPUT_DRIFT


def cmd_prune(store: RunStore, keep: int) -> int:
    runs = store.list_runs()
    if len(runs) <= keep:
        print(f"{len(runs)} runs, nothing to prune (keep={keep})")
        return 0
    for run_dir in runs[:-keep]:
        shutil.rmtree(run_dir)
        print(f"pruned {run_dir.name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prices = load_prices(args.prices)
    store = RunStore(args.root)
    if args.command == "ls":
        return cmd_ls(store, prices, args.as_json)
    if args.command == "report":
        return cmd_report(store, prices, args.run)
    if args.command == "diff":
        try:
            return cmd_diff(store, prices, args.a, args.b, args.context, args.as_json)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_STRUCTURE_DRIFT
    if args.command == "prune":
        return cmd_prune(store, args.keep)
    return 2


if __name__ == "__main__":
    sys.exit(main())
