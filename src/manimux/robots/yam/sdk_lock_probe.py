"""Opt-in, finite SDK lock trace. The ordinary collection launcher is unaffected.

Run with --check for a hardware-free source audit, or pass collection arguments
after -- to launch the existing GUI. Capture starts at the first active normal
teleop cycle. No SDK file, lock object, control rate or getter result is replaced.
Only SDK method *with contexts* are instrumented in the diagnostic process.
"""

from __future__ import annotations

import argparse
import ast
import bisect
import copy
import difflib
import hashlib
import importlib
import inspect
import json
import math
import sys
import textwrap
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from manimux.timing import _prefix, active_timing

SDK_SOURCES = {
    "i2rt.robots.motor_chain_robot": (
        "MotorChainRobot",
        "755dfd973c46b20d42a68cba74ac81156d6ec994b6ab2026ae25309bc62b28bc",
    ),
    "i2rt.motor_drivers.dm_driver": (
        "DMChainCanInterface",
        "8d74b3a48bc883bf7e1578906b4004666cbc1e2a2c496f4528e294546b7fa738",
    ),
}
LOCKS = {"_state_lock", "_command_lock", "state_lock", "command_lock", "same_bus_device_lock"}
CALLS = {"get_joint_pos", "_compute_gravity_compensation", "_set_commands"}
HELPER = "_manimux_sdk_lock_probe"
FIELDS = (
    "span_id", "parent_id", "site", "kind", "channel", "lock_id", "cycle_ns",
    "stage_prefix", "start_ns", "acquired_ns", "body_end_ns", "released_ns",
    "wait_cpu_ns", "hold_cpu_ns", "error",
)


class InstrumentLocks(ast.NodeTransformer):
    def __init__(self, site):
        self.site = site
        self.count = 0

    def visit_With(self, node):
        self.generic_visit(node)
        for item in node.items:
            expr = item.context_expr
            if (isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name)
                    and expr.value.id == "self" and expr.attr in LOCKS):
                item.context_expr = ast.copy_location(ast.Call(
                    func=ast.Attribute(value=ast.Name(id=HELPER, ctx=ast.Load()),
                                       attr="lock", ctx=ast.Load()),
                    args=[ast.Name(id="self", ctx=ast.Load()), expr,
                          ast.Constant(self.site + "." + expr.attr)], keywords=[],
                ), expr)
                self.count += 1
        return node


def instrument_function(function, class_name):
    source = textwrap.dedent(inspect.getsource(function))
    tree = ast.parse(source)
    original = copy.deepcopy(tree)
    transform = InstrumentLocks(class_name + "." + function.__name__)
    transform.visit(tree)
    if function.__name__ in CALLS:
        fn = tree.body[0]
        # Preserve the function docstring as a docstring.
        head = 1 if (isinstance(fn.body[0], ast.Expr)
                     and isinstance(fn.body[0].value, ast.Constant)
                     and isinstance(fn.body[0].value.value, str)) else 0
        body = fn.body[head:]
        context = ast.Call(
            func=ast.Attribute(value=ast.Name(id=HELPER, ctx=ast.Load()),
                               attr="call", ctx=ast.Load()),
            args=[ast.Name(id="self", ctx=ast.Load()),
                  ast.Constant(class_name + "." + function.__name__)], keywords=[],
        )
        fn.body[head:] = [ast.copy_location(ast.With(
            items=[ast.withitem(context_expr=context)], body=body,
        ), body[0])]
    ast.fix_missing_locations(tree)
    # Fail before launch if anything except the diagnostic contexts changed.
    restored = strip_instrumentation(copy.deepcopy(tree))
    if ast.dump(restored) != ast.dump(original):
        raise RuntimeError(f"probe changed non-timing statements in {function.__qualname__}")
    return tree, transform.count, source


def strip_instrumentation(tree):
    class Strip(ast.NodeTransformer):
        def visit_With(self, node):
            self.generic_visit(node)
            for item in node.items:
                expr = item.context_expr
                if (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute)
                        and isinstance(expr.func.value, ast.Name)
                        and expr.func.value.id == HELPER):
                    if expr.func.attr == "call":
                        return node.body
                    item.context_expr = expr.args[1]
            return node
    return Strip().visit(tree)


class ThreadBuffer:
    def __init__(self, capacity):
        self.name = threading.current_thread().name
        self.native_id = threading.get_native_id()
        self.events = deque(maxlen=capacity)
        self.stack = []
        self.next_id = 0
        self.completed = 0


class Span:
    def __init__(self, probe, owner, lock, site):
        self.probe, self.owner, self.original, self.site = probe, owner, lock, site
        self.buffer = None

    def __enter__(self):
        probe = self.probe
        row = active_timing.get()
        if (probe.started_ns is None and self.site == "MotorChainRobot.get_joint_pos"
                and row and row.get("kind") == "cycle" and row.get("sync_enabled")):
            probe.arm()
        if probe.capturing and time.monotonic_ns() < probe.deadline_ns:
            buf = getattr(probe.local, "buffer", None)
            if buf is None:
                buf = ThreadBuffer(probe.capacity)
                probe.local.buffer = buf
                probe.buffers.append(buf)
            self.buffer = buf
            self.parent = buf.stack[-1] if buf.stack else None
            buf.next_id += 1
            self.span_id = buf.next_id
            buf.stack.append(self.span_id)
            chain = getattr(self.owner, "motor_chain", self.owner)
            self.channel = getattr(chain, "channel", "unknown")
            self.cycle_ns = row.get("cycle_monotonic_ns") if row else None
            self.prefix = _prefix.get()
            self.cpu_start = time.thread_time_ns()
            self.start_ns = time.monotonic_ns()
        try:
            result = self.original.__enter__() if self.original is not None else None
        except BaseException:
            if self.buffer is not None:
                self.buffer.stack.pop()
            raise
        if self.buffer is not None:
            self.acquired_ns = time.monotonic_ns()
            self.cpu_acquired = time.thread_time_ns()
        return result

    def __exit__(self, exc_type, exc, tb):
        buf = self.buffer
        if buf is not None:
            body_end = time.monotonic_ns()
            cpu_end = time.thread_time_ns()
        try:
            # Delegate the original context exactly once, even on body exceptions.
            return self.original.__exit__(exc_type, exc, tb) if self.original is not None else False
        finally:
            if buf is not None:
                released = time.monotonic_ns()
                buf.stack.pop()
                if self.probe.capturing:
                    buf.completed += 1
                    buf.events.append((
                        self.span_id, self.parent, self.site,
                        "lock" if self.original is not None else "call", self.channel,
                        id(self.original) if self.original is not None else None,
                        self.cycle_ns, self.prefix, self.start_ns, self.acquired_ns,
                        body_end, released, self.cpu_acquired - self.cpu_start,
                        cpu_end - self.cpu_acquired, exc_type.__name__ if exc_type else None,
                    ))


class LockProbe:
    def __init__(self, output, seconds=30.0, capacity=100_000):
        if not math.isfinite(seconds) or seconds <= 0 or capacity < 1:
            raise ValueError("positive seconds and buffer capacity required")
        if sys.implementation.name != "cpython":
            raise RuntimeError("probe buffers require CPython's atomic deque append/snapshot")
        self.output = Path(output)
        self.seconds, self.capacity = seconds, capacity
        self.local = threading.local()
        self.buffers = deque()
        self.started_ns = None
        self.deadline_ns = 0
        self.capturing = False
        self.trigger = threading.Event()
        self.stop = threading.Event()
        self.worker = None
        self.error = None

    def arm(self):
        if self.started_ns is None:
            self.started_ns = time.monotonic_ns()
            self.deadline_ns = self.started_ns + round(self.seconds * 1e9)
            self.capturing = True
            self.trigger.set()

    def lock(self, owner, lock, site):
        return Span(self, owner, lock, site)

    def call(self, owner, site):
        return Span(self, owner, None, site)

    def start_writer(self):
        self.worker = threading.Thread(target=self._write_after_capture,
                                       name="sdk-lock-probe-save", daemon=True)
        self.worker.start()

    def _write_after_capture(self):
        self.trigger.wait()
        if self.started_ns is not None:
            remaining = max(0, (self.deadline_ns - time.monotonic_ns()) / 1e9)
            self.stop.wait(remaining)
            # Let spans that started before the deadline finish, with a finite grace.
            if not self.stop.is_set():
                self.stop.wait(0.5)
        self.capturing = False
        try:
            self.save()
            print(f"[sdk-lock-probe] saved: {self.output / 'summary.json'}", flush=True)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"[sdk-lock-probe] SAVE ERROR: {self.error}", file=sys.stderr, flush=True)

    def close(self):
        self.stop.set()
        self.trigger.set()
        if self.worker is not None:
            self.worker.join(timeout=20)
            if self.worker.is_alive():
                print("[sdk-lock-probe] save still running; output may be incomplete",
                      file=sys.stderr, flush=True)

    def save(self):
        self.output.mkdir(parents=True, exist_ok=True)
        rows, threads = [], []
        for buf in list(self.buffers):
            events = list(buf.events)
            threads.append({"thread_id": buf.native_id, "thread_name": buf.name,
                            "completed": buf.completed, "retained": len(events),
                            "dropped": buf.completed - len(events),
                            "in_flight_at_end": list(buf.stack)})
            for event in events:
                rows.append({"thread_id": buf.native_id, "thread_name": buf.name,
                             **dict(zip(FIELDS, event, strict=True))})
        rows.sort(key=lambda r: r["start_ns"])
        pending = self.output / "sdk-locks.jsonl.part"
        with pending.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        pending.rename(self.output / "sdk-locks.jsonl")
        report = summarize(rows)
        report.update({
            "started_ns": self.started_ns, "deadline_ns": self.deadline_ns,
            "requested_seconds": self.seconds, "threads": threads,
            "scope": "lock acquisition interval includes scheduling/GIL delay; hold is wall "
                     "time while the original lock is owned; nested spans overlap; lock_id "
                     "is per object; cycle_ns joins control-timing.jsonl; finite probe adds "
                     "overhead; uncompleted boundary spans are listed, not estimated",
        })
        (self.output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (self.output / "write_complete.flag").touch()


def stats(values):
    values = sorted(values)
    return {"count": len(values), "mean_ms": sum(values) / len(values) / 1e6,
            "p95_ms": values[max(0, math.ceil(0.95 * len(values)) - 1)] / 1e6,
            "max_ms": values[-1] / 1e6}


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["channel"], row["site"], row["thread_name"], row["thread_id"])].append(row)
    report = []
    for (channel, site, thread, tid), events in sorted(groups.items()):
        report.append({
            "channel": channel, "site": site, "thread_name": thread, "thread_id": tid,
            "kind": events[0]["kind"],
            "wait": stats([r["acquired_ns"] - r["start_ns"] for r in events]),
            "hold": stats([r["body_end_ns"] - r["acquired_ns"] for r in events]),
            "total": stats([r["released_ns"] - r["start_ns"] for r in events]),
            "wait_cpu": stats([r["wait_cpu_ns"] for r in events]),
            "hold_cpu": stats([r["hold_cpu_ns"] for r in events]),
        })
    return {"rows": len(rows), "groups": report, **getter_breakdown(rows)}


def getter_breakdown(rows):
    """Pair each complete getter with its exact acquire span, never sum P95s."""
    by_parent = defaultdict(list)
    holders = defaultdict(list)
    for row in rows:
        by_parent[(row["thread_id"], row["parent_id"])].append(row)
        if row["kind"] == "lock":
            holders[row["lock_id"]].append(row)
    for intervals in holders.values():
        intervals.sort(key=lambda r: r["acquired_ns"])
    holder_starts = {key: [r["acquired_ns"] for r in events]
                     for key, events in holders.items()}
    samples = []
    for row in rows:
        if row["site"] != "MotorChainRobot.get_joint_pos":
            continue
        children = by_parent[(row["thread_id"], row["span_id"])]
        acquire = next((r for r in children
                        if r["site"] == "MotorChainRobot.get_joint_pos._state_lock"), None)
        if acquire is None:
            continue
        total = row["released_ns"] - row["start_ns"]
        wait = acquire["acquired_ns"] - acquire["start_ns"]
        hold = acquire["body_end_ns"] - acquire["acquired_ns"]
        samples.append({
            "channel": row["channel"], "thread_id": row["thread_id"],
            "cycle_ns": row["cycle_ns"], "stage_prefix": row["stage_prefix"],
            "start_ns": row["start_ns"], "total_ns": total, "wait_ns": wait,
            "hold_ns": hold, "outside_lock_ns": total - wait - hold,
            "wait_cpu_ns": acquire["wait_cpu_ns"], "lock_id": acquire["lock_id"],
            "wait_start_ns": acquire["start_ns"], "wait_end_ns": acquire["acquired_ns"],
        })
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["channel"]].append(sample)
    breakdown = []
    for channel, data in sorted(grouped.items()):
        breakdown.append({"channel": channel, "paired_getters": len(data),
                          **{key.removesuffix("_ns"): stats([s[key] for s in data])
                             for key in ("total_ns", "wait_ns", "hold_ns", "outside_lock_ns")},
                          "acquire_interval_fraction": (
                              sum(s["wait_ns"] for s in data) / sum(s["total_ns"] for s in data)
                          )})
    slowest = sorted(samples, key=lambda s: s["total_ns"], reverse=True)[:20]
    for sample in slowest:
        start, end = sample["wait_start_ns"], sample["wait_end_ns"]
        candidates = holders[sample["lock_id"]]
        # These SDK state locks are non-reentrant; at most one holder precedes start.
        left = max(0, bisect.bisect_left(holder_starts[sample["lock_id"]], start) - 1)
        right = bisect.bisect_right(holder_starts[sample["lock_id"]], end)
        overlaps = []
        for owner in candidates[left:right]:
            overlap = min(end, owner["body_end_ns"]) - max(start, owner["acquired_ns"])
            if overlap <= 0 or owner["thread_id"] == sample["thread_id"]:
                continue
            children = by_parent[(owner["thread_id"], owner["span_id"])]
            overlaps.append({
                "site": owner["site"], "thread_id": owner["thread_id"],
                "overlap_ns": overlap, "hold_ns": owner["body_end_ns"] - owner["acquired_ns"],
                "hold_cpu_ns": owner["hold_cpu_ns"],
                "nested_spans": [{"site": c["site"], "kind": c["kind"],
                                  "wait_ns": c["acquired_ns"] - c["start_ns"],
                                  "hold_ns": c["body_end_ns"] - c["acquired_ns"]}
                                 for c in children],
            })
        sample["observed_holders_during_wait"] = overlaps
        sample["observed_holder_overlap_ns"] = sum(o["overlap_ns"] for o in overlaps)
    return {"getter_breakdown": breakdown, "slowest_getters": slowest}


def prepare(probe):
    """Import classes only; never construct robots/CAN interfaces or start SDK threads."""
    plan, sources, diffs = [], [], []
    for module_name, (class_name, expected_sha) in SDK_SOURCES.items():
        spec = importlib.util.find_spec(module_name)
        path = Path(spec.origin)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != expected_sha:
            raise RuntimeError(f"SDK source changed; re-audit before tracing: {path} ({sha})")
        module = importlib.import_module(module_name)
        if HELPER in module.__dict__:
            raise RuntimeError("SDK lock probe already installed in this process")
        cls = getattr(module, class_name)
        coverage = {}
        for name, function in vars(cls).items():
            if name == "__init__" or not inspect.isfunction(function):
                continue
            tree, count, source = instrument_function(function, class_name)
            if not count and name not in CALLS:
                continue
            coverage[name] = count
            modified = ast.unparse(tree) + "\n"
            diffs.extend(difflib.unified_diff(source.splitlines(True), modified.splitlines(True),
                         fromfile=f"original/{class_name}.{name}",
                         tofile=f"instrumented/{class_name}.{name}"))
            ast.increment_lineno(tree, function.__code__.co_firstlineno - 1)
            namespace = {}
            exec(compile(tree, str(path), "exec"), module.__dict__, namespace)
            instrumented = namespace[name]
            instrumented.__qualname__ = function.__qualname__
            instrumented.__annotations__ = function.__annotations__
            plan.append((module, cls, name, function, instrumented))
        sources.append({"module": module_name, "path": str(path), "sha256": sha,
                        "lock_context_counts": coverage})
    probe.output.mkdir(parents=True, exist_ok=False)
    (probe.output / "instrumentation.patch").write_text("".join(diffs), encoding="utf-8")
    app_root = Path(__file__).resolve().parents[2]
    app_paths = (
        "timing.py", "robots/yam/driver.py", "robots/yam/base.py", "robots/yam/arm.py",
        "collection/yam/backend.py", "collection/yam/teleop/loop.py",
        "collection/yam/robot/yam_adapter.py", "collection/yam/config.py",
    )
    app_sources = {name: hashlib.sha256((app_root / name).read_bytes()).hexdigest()
                   for name in app_paths}
    (probe.output / "manifest.json").write_text(json.dumps({
        "sources": sources, "argv": sys.argv, "seconds": probe.seconds,
        "application_source_sha256": app_sources,
        "capacity_per_thread": probe.capacity,
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "trigger": "first SDK getter in active normal teleop cycle; alignment excluded",
        "disk_changes_to_sdk": False, "lock_objects_replaced": False,
        "control_statements_unchanged_ast_check": True,
    }, indent=2), encoding="utf-8")
    return plan


def install(probe, plan):
    for module, cls, name, _original, modified in plan:
        module.__dict__[HELPER] = probe
        setattr(cls, name, modified)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    split = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--check", action="store_true", help="source audit only; no GUI or devices")
    args = parser.parse_args(argv[:split])
    collect_args = argv[split + 1:]
    if not args.check and not collect_args:
        parser.error("pass the existing collection arguments after --")
    probe = LockProbe(args.output, seconds=args.seconds)
    plan = prepare(probe)
    print(f"[sdk-lock-probe] source audit passed; {len(plan)} methods; {probe.output}", flush=True)
    if args.check:
        return
    install(probe, plan)
    probe.start_writer()
    print(f"[sdk-lock-probe] waiting for active teleop; will capture {args.seconds:g} seconds",
          flush=True)
    try:
        from manimux.collection.cli import main as collect
        collect(collect_args)
    finally:
        probe.close()


if __name__ == "__main__":
    main()
