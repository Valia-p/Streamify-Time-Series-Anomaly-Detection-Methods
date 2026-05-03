from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np

from stream_loader import StreamData


class StreamingOrchestrator:
    def __init__(self, stream_data: StreamData, batch_size: int = 1000):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.stream_data = stream_data
        self.batch_size = int(batch_size)

    def iter_batches(self) -> Iterator[Tuple[int, int, int, np.ndarray, np.ndarray]]:
        values, labels = self.get_full_arrays()
        n = len(values)
        batch_idx = 0
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            yield batch_idx, start, end, values[start:end], labels[start:end]
            batch_idx += 1

    def total_batches(self) -> int:
        n = len(self.stream_data.values)
        return int((n + self.batch_size - 1) // self.batch_size)

    def get_full_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.stream_data.values, self.stream_data.labels


@dataclass
class _BatchTimer:
    _start: float | None = None
    elapsed: float = 0.0

    def __enter__(self) -> "_BatchTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        end = time.perf_counter()
        if self._start is None:
            self.elapsed = 0.0
        else:
            self.elapsed = float(end - self._start)


class LatencyTracker:
    def __init__(self) -> None:
        self._batch_indices: List[int] = []
        self._elapsed: List[float] = []

    def time_batch(self) -> _BatchTimer:
        return _BatchTimer()

    def record(self, batch_idx: int, elapsed_seconds: float) -> None:
        self._batch_indices.append(int(batch_idx))
        self._elapsed.append(float(elapsed_seconds))

    def summary(self) -> dict:
        if not self._elapsed:
            return {
                "total_time": 0.0,
                "mean_latency": 0.0,
                "median_latency": 0.0,
                "p95_latency": 0.0,
                "max_latency": 0.0,
                "n_batches": 0,
            }

        arr = np.asarray(self._elapsed, dtype=float)
        return {
            "total_time": float(np.sum(arr)),
            "mean_latency": float(np.mean(arr)),
            "median_latency": float(np.median(arr)),
            "p95_latency": float(np.percentile(arr, 95)),
            "max_latency": float(np.max(arr)),
            "n_batches": int(arr.size),
        }
