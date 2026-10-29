"""tapedeck -- a flight recorder for LLM agents.

Record a live agent run once, then replay it offline forever: deterministic,
free, CI-friendly. Diff any two runs to find the first step where behavior
drifted, with token and cost accounting per model.

    import tapedeck

    with tapedeck.record("chat-v3") as tape:
        answer = my_agent(tape, user_message)      # every tape.call is traced

    # ... later, on a changed codebase, in CI, with zero API spend:
    with tapedeck.play("chat-v3") as tape:
        answer = my_agent(tape, user_message)      # serves recorded outputs
"""

from .costs import total_cost
from .diff import diff_runs
from .redact import Redactor
from .session import Event, RecordingTape, ReplayDrift, ReplayingTape
from .store import DEFAULT_ROOT, RunStore

__version__ = "0.2.0"

__all__ = [
    "record", "play", "load_run", "Redactor", "RecordingTape", "ReplayingTape",
    "ReplayDrift", "Event", "RunStore", "diff_runs", "total_cost",
    "DEFAULT_ROOT", "__version__",
]


def record(name: str, root=None, redact: Redactor | None = None) -> RecordingTape:
    """Start recording a run named ``name`` into ``root`` (default .tapedeck/runs).

    Usable as a context manager -- the run is saved on clean exit::

        with tapedeck.record("chat-v3") as tape:
            my_agent(tape, message)
    """
    store = RunStore(root)
    run_dir = store.new_run_dir(name)
    tape = RecordingTape(redact=redact)
    tape._run_dir = run_dir
    tape._name = name
    tape._root = root
    tape._saved = False
    return tape


def play(ref: str, root=None, strict: bool = True, lenient: bool = False) -> ReplayingTape:
    """Load a recorded run (exact dir, path, or unique name prefix) for replay.

    Also a context manager; on exit it verifies every recorded step was
    reached, so ``with tapedeck.play("run") as tape:`` *is* the regression test.
    """
    store = RunStore(root)
    run_dir = store.resolve(ref)
    events, _meta = RunStore.read_run(run_dir)
    return ReplayingTape(events=events, strict=strict, lenient=lenient)


def save(tape: RecordingTape, root=None, meta_extra: dict | None = None):
    """Persist a finished recording session to disk."""
    if getattr(tape, "_saved", False):
        return tape._run_dir
    store = RunStore(root if root is not None else getattr(tape, "_root", None))
    tape._saved = True
    return store.write_run(tape._run_dir, tape, tape._name, meta_extra)


def load_run(ref: str, root=None):
    """Read one run as (events, meta) -- the data behind ls/report/diff."""
    store = RunStore(root)
    return RunStore.read_run(store.resolve(ref))
