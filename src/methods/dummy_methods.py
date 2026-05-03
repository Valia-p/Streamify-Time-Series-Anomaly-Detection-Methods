"""Dummy methods used for scaffold smoke testing."""

from __future__ import annotations

import numpy as np

from streaming_orchestrator import LatencyTracker, StreamingOrchestrator


def run_dummy_random_offline(values: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random(len(values))


def run_dummy_random_streaming(
    orchestrator: StreamingOrchestrator, tracker: LatencyTracker
) -> np.ndarray:
    rng = np.random.default_rng(42)
    scores = np.zeros(len(orchestrator.stream_data.values), dtype=float)
    for batch_idx, start, end, _vals, _labels in orchestrator.iter_batches():
        with tracker.time_batch() as t:
            scores[start:end] = rng.random(end - start)
        tracker.record(batch_idx, t.elapsed)
    return scores
