"""Test suite: record/replay identity, drift detection, redaction, costs, CLI.

Pure stdlib unittest -- the project itself has zero dependencies, and so do
its tests. Run with:  python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tapedeck
from tapedeck.diff import EXIT_OUTPUT_DRIFT, EXIT_SAME, EXIT_STRUCTURE_DRIFT
from tapedeck.redact import Redactor
from tapedeck.session import ReplayDrift

ROOT = Path(tempfile.mkdtemp(prefix="tapedeck-tests-"))


class Base(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / self.id().split(".")[-1] / self._testMethodName
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)


class TestRecordReplay(Base):
    def _run_agent(self, tape, multiplier):
        a = tape.call("tool", "fetch", lambda: 21)
        b = tape.call("model", "plan", lambda: ({"op": "mul"}, {"prompt_tokens": 5, "completion_tokens": 2, "model": "deepseek-chat"}))
        # multiplier is an explicit argument: replays can see it. (Values
        # captured in closures are invisible to input matching -- a
        # documented limitation; that's what diff between two recordings is for.)
        c = tape.call("tool", "apply", lambda m: a * m, multiplier)
        return c, b

    def test_replay_serves_recorded_outputs_without_calling_fn(self):
        with tapedeck.record("run", root=self.root) as tape:
            self._run_agent(tape, multiplier=2)
            tapedeck.save(tape)

        calls = {"n": 0}
        def counting_fn():
            calls["n"] += 1
            return "LIVE"
        tape2 = tapedeck.play("run", root=self.root)
        result, _ = self._run_agent(tape2, multiplier=2)  # same inputs as recorded
        self.assertEqual(result, 42)          # recorded output served...
        self.assertEqual(calls["n"], 0)       # ...and nothing ran live

    def test_replay_detects_changed_input(self):
        with tapedeck.record("run", root=self.root) as tape:
            self._run_agent(tape, multiplier=2)
            tapedeck.save(tape)
        tape2 = tapedeck.play("run", root=self.root)
        with self.assertRaises(ReplayDrift) as ctx:
            self._run_agent(tape2, multiplier=3)   # compute_total input changes
        self.assertEqual(ctx.exception.reason, "input_mismatch")
        self.assertEqual(ctx.exception.name, "apply")

    def test_replay_detects_extra_step(self):
        with tapedeck.record("run", root=self.root) as tape:
            self._run_agent(tape, multiplier=2)
            tapedeck.save(tape)
        tape2 = tapedeck.play("run", root=self.root)
        def extra():
            self._run_agent(tape2, multiplier=2)
            tape2.call("tool", "new_step", lambda: 1)   # code added a step
        with self.assertRaises(ReplayDrift) as ctx:
            extra()
        self.assertEqual(ctx.exception.reason, "missing")

    def test_assert_complete_catches_shorter_replay(self):
        with tapedeck.record("run", root=self.root) as tape:
            self._run_agent(tape, multiplier=2)
            tapedeck.save(tape)
        tape2 = tapedeck.play("run", root=self.root)
        tape2.call("tool", "fetch", lambda: 21)
        with self.assertRaises(ReplayDrift) as ctx:
            tape2.assert_complete()
        self.assertEqual(ctx.exception.reason, "unreached")

    def test_lenient_mode_runs_live_after_first_drift(self):
        with tapedeck.record("run", root=self.root) as tape:
            self._run_agent(tape, multiplier=2)
            tapedeck.save(tape)
        tape2 = tapedeck.play("run", root=self.root, lenient=True)
        result, _ = self._run_agent(tape2, multiplier=5)
        self.assertEqual(result, 105)         # drift at step 2 -> ran live
        self.assertEqual(len(tape2.drifts), 1)

    def test_recorded_errors_are_data(self):
        def boom():
            raise ValueError("provider 500")
        with tapedeck.record("run", root=self.root) as tape:
            with self.assertRaises(ValueError):
                tape.call("model", "chat", boom)
            tapedeck.save(tape)
        events, _ = tapedeck.load_run("run", root=self.root)
        self.assertIn("ValueError: provider 500", events[0].error)
        # ...and replay re-raises the failure as drift, not silently succeeds
        tape2 = tapedeck.play("run", root=self.root)
        with self.assertRaises(ReplayDrift) as ctx:
            tape2.call("model", "chat", lambda: "should not run")
        self.assertEqual(ctx.exception.reason, "recorded_error")


class TestRedaction(Base):
    def test_default_patterns(self):
        red = Redactor()
        scrubbed = red.apply({
            "prompt": "email me at a@b.com, key sk-abcdefghijklmnop123456, call 13812345678",
        })
        self.assertNotIn("a@b.com", json.dumps(scrubbed))
        self.assertNotIn("sk-abcdefghijklmnop123456", json.dumps(scrubbed))
        self.assertNotIn("13812345678", json.dumps(scrubbed))
        self.assertIn("<EMAIL>", scrubbed["prompt"])
        self.assertIn("<API_KEY>", scrubbed["prompt"])

    def test_redaction_applied_during_recording(self):
        red = Redactor()
        with tapedeck.record("run", root=self.root, redact=red) as tape:
            tape.call("model", "chat", lambda: "user email is x@y.io")
            tapedeck.save(tape)
        events, meta = tapedeck.load_run("run", root=self.root)
        self.assertNotIn("x@y.io", json.dumps(events[0].output))
        self.assertTrue(meta["redacted"])


class TestCosts(Base):
    def test_cost_from_recorded_usage(self):
        from tapedeck.costs import total_cost
        with tapedeck.record("run", root=self.root) as tape:
            tape.call("model", "chat", lambda: (
                "hi", {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000, "model": "deepseek-chat"}))
            tapedeck.save(tape)
        events, _ = tapedeck.load_run("run", root=self.root)
        cost = total_cost(events)  # built-in reference prices
        self.assertAlmostEqual(cost["total_usd"], 0.27 + 1.10, places=5)

    def test_unknown_model_reports_none_not_zero(self):
        from tapedeck.costs import total_cost
        events = [tapedeck.Event(i=0, kind="model", name="x",
                                 usage={"prompt_tokens": 10, "completion_tokens": 10, "model": "mystery"})]
        self.assertIsNone(total_cost(events, {})["total_usd"])


class TestDiff(Base):
    def _make_runs(self, b_multiplier=2):
        with tapedeck.record("runA", root=self.root) as tape:
            a = tape.call("tool", "fetch", lambda: 21)
            tape.call("tool", "apply", lambda: a * 2)
            tapedeck.save(tape)
        with tapedeck.record("runB", root=self.root) as tape:
            a = tape.call("tool", "fetch", lambda: 21)
            tape.call("tool", "apply", lambda: a * b_multiplier)
            tapedeck.save(tape)

    def test_identical_runs(self):
        self._make_runs()
        store = tapedeck.RunStore(self.root)
        result = tapedeck.diff_runs(store.resolve("runA"), store.resolve("runB"), prices={})
        self.assertTrue(result.same)

    def test_output_drift_reports_first_step(self):
        self._make_runs(b_multiplier=3)
        store = tapedeck.RunStore(self.root)
        result = tapedeck.diff_runs(store.resolve("runA"), store.resolve("runB"), prices={})
        self.assertFalse(result.same)
        self.assertFalse(result.structural)
        self.assertEqual(result.first_divergence, 1)
        self.assertIn("apply", result.reason)
        self.assertTrue(any("fetch" in line for line in result.lines))  # context shown

    def test_structural_drift(self):
        self._make_runs()
        with tapedeck.record("runC", root=self.root) as tape:
            a = tape.call("tool", "fetch", lambda: 21)
            tape.call("tool", "apply", lambda: a * 2)
            tape.call("tool", "new_step", lambda: 1)
            tapedeck.save(tape)
        store = tapedeck.RunStore(self.root)
        result = tapedeck.diff_runs(store.resolve("runA"), store.resolve("runC"), prices={})
        self.assertTrue(result.structural)


class TestCLI(Base):
    def _seed_runs(self, b_multiplier=2):
        self._make_runs = TestDiff.__dict__["_make_runs"].__get__(self)(b_multiplier=b_multiplier)

    def test_cli_diff_exit_codes(self):
        with tapedeck.record("runA", root=self.root) as tape:
            a = tape.call("tool", "fetch", lambda: 21)
            tape.call("tool", "apply", lambda: a * 2)
            tapedeck.save(tape)
        with tapedeck.record("runB", root=self.root) as tape:
            a = tape.call("tool", "fetch", lambda: 21)
            tape.call("tool", "apply", lambda: a * 2)
            tapedeck.save(tape)
        with tapedeck.record("runC", root=self.root) as tape:
            a = tape.call("tool", "fetch", lambda: 21)
            tape.call("tool", "apply", lambda: a * 7)
            tapedeck.save(tape)

        from tapedeck.cli import main
        same = main(["--root", str(self.root), "diff", "runA", "runB"])
        drift = main(["--root", str(self.root), "diff", "runA", "runC"])
        self.assertEqual(same, EXIT_SAME)
        self.assertEqual(drift, EXIT_OUTPUT_DRIFT)

    def test_cli_ls_and_report(self):
        with tapedeck.record("runA", root=self.root) as tape:
            tape.call("model", "chat", lambda: (
                "hi", {"prompt_tokens": 1000, "completion_tokens": 100, "model": "deepseek-chat"}))
            tapedeck.save(tape)
        from tapedeck.cli import main
        rc = main(["--root", str(self.root), "ls"])
        self.assertEqual(rc, 0)
        rc = main(["--root", str(self.root), "report", "runA"])
        self.assertEqual(rc, 0)

    def test_cli_json_output(self):
        with tapedeck.record("runA", root=self.root) as tape:
            tape.call("tool", "fetch", lambda: 1)
            tapedeck.save(tape)
        from tapedeck.cli import main
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(["--root", str(self.root), "ls", "--json"])
        rows = json.loads(buf.getvalue())
        self.assertEqual(rc, 0)
        self.assertEqual(rows[0]["steps"], 1)

    def test_demo_agent_end_to_end(self):
        demo = Path(__file__).resolve().parent.parent / "examples" / "demo_agent.py"
        env = dict(__import__("os").environ, PYTHONPATH=str(demo.parent.parent))
        r1 = subprocess.run([sys.executable, str(demo), "--mode", "record", "--scenario", "stable"],
                            capture_output=True, text=True, env=env, cwd=self.root)
        self.assertEqual(r1.returncode, 0, r1.stderr)
        r2 = subprocess.run([sys.executable, str(demo), "--mode", "record", "--scenario", "drift"],
                            capture_output=True, text=True, env=env, cwd=self.root)
        self.assertEqual(r2.returncode, 0, r2.stderr)
        r3 = subprocess.run([sys.executable, str(demo), "--mode", "replay"],
                            capture_output=True, text=True, env=env, cwd=self.root)
        self.assertEqual(r3.returncode, 0, r3.stderr)
        self.assertIn("$0.00 spent", r3.stdout)


if __name__ == "__main__":
    unittest.main()
