"""Bounded dual-arm request/response transport for the peer RL-IK CPU workers."""

import os
import pickle
import select
import socket
import subprocess
import time
from collections import deque
from pathlib import Path

import numpy as np

from manimux.kinematics.poses import pose_matrix

ARMS = ("left_arm", "right_arm")
ROOT = Path(__file__).resolve().parents[2]


class DualArmRefiner:
    def __init__(self, options):
        self.options = options
        if set(options) != {"python", "merge_reserve_us", *ARMS}:
            raise ValueError("rl_ik requires python, merge_reserve_us and both arm exports")
        self.reserve_ns = int(options["merge_reserve_us"] * 1000)
        if not 0 < self.reserve_ns < 2_000_000:
            raise ValueError("invalid RL merge reserve")
        self.workers = {}
        self.sequence = 0
        self.history = deque(maxlen=8)
        self.pending = {}
        self.identities = {}

    def close(self):
        for channel, process in self.workers.values():
            channel.close()
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        self.workers.clear()
        self.pending.clear()
        self.history.clear()

    def warmup(self, models, groups):
        if set(groups) != set(ARMS):
            raise ValueError("RL step IK requires both Tianji arms")
        if any(proc.poll() is not None for _, proc in self.workers.values()):
            self.close()
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OPENBLAS_NUM_THREADS": "1",
               "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        try:
            for name in ARMS:
                if name not in self.workers:
                    parent, child = socket.socketpair(type=socket.SOCK_SEQPACKET)
                    spec = self.options[name]
                    try:
                        proc = subprocess.Popen(
                            [ROOT / self.options["python"], "-m", "rl_ik.step_worker",
                             str(child.fileno()), ROOT / spec["actor"], spec["sha256"],
                             name.removesuffix("_arm")], env=env, pass_fds=(child.fileno(),),
                        )
                    except BaseException:
                        parent.close()
                        raise
                    finally:
                        child.close()
                    self.workers[name] = (parent, proc)
                channel, _ = self.workers[name]
                channel.settimeout(15)
                channel.send(pickle.dumps(("warm", groups[name][:7])))
            for name in ARMS:
                channel, _ = self.workers[name]
                # A timed-out result can still precede the warmup reply.
                while True:
                    packet = channel.recv(8192)
                    if not packet:
                        raise ValueError(f"{name} worker exited during artifact or FK validation")
                    reply = pickle.loads(packet)
                    if isinstance(reply[0], str):
                        break
                side, pose, pid, source = reply
                if side + "_arm" != name or not np.allclose(
                    pose, models[name].fk(groups[name]), atol=1e-7, rtol=0,
                ):
                    raise ValueError(f"{name} policy TCP or arm mismatch")
                self.identities[name] = {"pid": pid, "source_sha256": source}
                channel.setblocking(False)
            self.history.clear()
            self.pending.clear()
        except BaseException:
            self.close()
            raise

    def drain_late(self, wait_s=0):
        """Recover real work counts by request ID, including at episode finish."""
        late = {}
        channels = {self.workers[name][0]: name for name in self.pending}
        if not channels:
            return late
        ready, _, _ = select.select(list(channels), [], [], wait_s)
        for channel in ready:
            name = channels[channel]
            seq, deadline = self.pending.pop(name)
            try:
                received, _, count, ms, end = pickle.loads(channel.recv(8192))
            except (EOFError, ConnectionError):
                late[name] = dict(sequence=seq, iterations=None, elapsed_ms=None,
                                  state="worker_lost")
                continue
            if seq != received:
                raise RuntimeError("RL IPC sequence mismatch")
            late[name] = dict(sequence=seq, iterations=count, elapsed_ms=ms,
                              timeout=True, worker_overran=end >= deadline)
        return late

    def finish_telemetry(self, wait_s):
        for _ in range(len(self.pending)):
            late = self.drain_late(wait_s=wait_s)
            if not late:
                break
            yield late

    def propose(self, reference, seed, measured, deadline):
        self.sequence += 1
        seq = self.sequence
        pair = {}
        late = self.drain_late()
        stats = {name: {"sequence": seq, "iterations": 0, "elapsed_ms": None,
                        "timeout": False, "state": "not_sent"} for name in ARMS}
        # Only previously issued input commands enter the 50 Hz history. A
        # worker's speculative poses exist for one request, including on failure.
        past = list(self.history)[-2::-2][::-1]
        self.history.append({name: seed[name][:7].copy() for name in ARMS})
        for name in ARMS:
            channel, proc = self.workers[name]
            if proc.poll() is not None:
                stats[name]["state"] = "worker_lost"
                continue
            if name in self.pending:
                stats[name].update(state="previous_request_busy", timeout=True)
                continue
            row = reference.groups[name][0]
            if not np.isfinite(row).all() or not 0 <= row[7] <= 1:
                stats[name]["state"] = "invalid_target"
                continue
            if time.monotonic_ns() >= deadline:
                stats[name].update(state="not_sent", timeout=True)
                continue
            request = (seq, deadline, self.reserve_ns, seed[name][:7], [p[name] for p in past],
                       float(measured[name][7]), pose_matrix(row[:7]))
            try:
                channel.send(pickle.dumps(request))
            except (BlockingIOError, ConnectionError):
                stats[name]["state"] = "worker_unavailable"
                continue
            self.pending[name] = (seq, deadline)
            stats[name].update(state="pending", iterations=None)
        waiting = {self.workers[n][0]: n for n in ARMS if self.pending.get(n) == (seq, deadline)}
        while waiting and (remaining := deadline - time.monotonic_ns()) > 0:
            ready, _, _ = select.select(list(waiting), [], [], remaining / 1e9)
            for channel in ready:
                name = waiting.pop(channel)
                del self.pending[name]
                try:
                    received, q, count, ms, end = pickle.loads(channel.recv(8192))
                except (EOFError, ConnectionError):
                    stats[name].update(state="worker_lost", iterations=None)
                    continue
                if received != seq:
                    raise RuntimeError("RL IPC sequence mismatch")
                expired = time.monotonic_ns() >= deadline or end >= deadline
                stats[name].update(iterations=count, elapsed_ms=ms, timeout=expired or count == 0,
                                   state="received" if count else "no_time_for_iteration")
                if count and not expired and q.shape == (7,) and np.isfinite(q).all():
                    pair[name] = q
        for name in waiting.values():
            stats[name].update(timeout=True)
        return pair, stats, late
