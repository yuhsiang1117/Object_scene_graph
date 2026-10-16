"""Per-module timing. Real-time performance is a headline claim of the paper,
so every pipeline stage is instrumented and reported separately from the
(asynchronous) LLM decision latency.
"""
from __future__ import annotations

import csv
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List


class Profiler:
    def __init__(self, sync: bool = False) -> None:
        self._samples: Dict[str, List[float]] = defaultdict(list)
        # Charge queued CUDA work to the section that queued it.
        #
        # Without this the timings are wall-clock around an ASYNCHRONOUS
        # queue: the detector returns as soon as its kernels are submitted,
        # and the stall lands on whichever later call happens to synchronise
        # -- usually the mover, which is not wrapped at all. Measured on
        # outputs/mf5_pass2_v17: 39% of the run sat inside the control loop
        # and outside every section. Off by default because synchronising
        # every section costs real time; turn it on to attribute, not to run.
        self._sync = bool(sync)

    def _synchronise(self) -> None:
        if not self._sync:
            return
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:  # noqa: BLE001 - timing must never break a run
            self._sync = False

    @contextmanager
    def timeit(self, name: str):
        self._synchronise()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._synchronise()
            self._samples[name].append(time.perf_counter() - t0)

    def add(self, name: str, seconds: float) -> None:
        self._samples[name].append(seconds)

    def report(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for name, xs in self._samples.items():
            arr = sorted(xs)
            n = len(arr)
            out[name] = {
                "count": n,
                "mean_ms": 1000.0 * sum(arr) / n,
                "median_ms": 1000.0 * arr[n // 2],
                "max_ms": 1000.0 * arr[-1],
                "total_s": sum(arr),
                # A real-time claim is about the tail, not the mean: the review
                # asked for latency, and a 1-in-10 slow frame is what a robot
                # feels. Nearest-rank, so the value is one actually measured.
                "p90_ms": 1000.0 * arr[min(n - 1, int(0.9 * n))],
                "p99_ms": 1000.0 * arr[min(n - 1, int(0.99 * n))],
            }
        return out

    def fps(self, name: str) -> float:
        xs = self._samples.get(name)
        if not xs:
            return 0.0
        return len(xs) / sum(xs)

    def write_csv(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        rep = self.report()
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["module", "count", "mean_ms", "median_ms", "max_ms", "total_s",
                        "p90_ms", "p99_ms"])
            for name, r in sorted(rep.items()):
                w.writerow(
                    [name, r["count"], f"{r['mean_ms']:.2f}", f"{r['median_ms']:.2f}",
                     f"{r['max_ms']:.2f}", f"{r['total_s']:.2f}",
                     f"{r['p90_ms']:.2f}", f"{r['p99_ms']:.2f}"]
                )
