"""Streaming SAND baseline.

This module will host the online SAND implementation for streaming batches.
Expected behavior:
- consume orchestrator batches
- incrementally update SAND model state
- emit one anomaly score per timestamp
- record per-batch latency in the provided tracker
"""

from __future__ import annotations

import numpy as np

from streaming_orchestrator import LatencyTracker, StreamingOrchestrator


def run_sand_streaming(
    orchestrator: StreamingOrchestrator, tracker: LatencyTracker
) -> np.ndarray:
    raise NotImplementedError("run_sand_streaming is not implemented yet.")
