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
    def __init__(self) -> None:
        self._samples: Dict[str, List[float]] = defaultdict(list)

    @contextmanager
    def timeit(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
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
            w.writerow(["module", "count", "mean_ms", "median_ms", "max_ms", "total_s"])
            for name, r in sorted(rep.items()):
                w.writerow(
                    [name, r["count"], f"{r['mean_ms']:.2f}", f"{r['median_ms']:.2f}",
                     f"{r['max_ms']:.2f}", f"{r['total_s']:.2f}"]
                )
