"""Variant 1: Naive batch-wise streaming Isolation Forest.

Expected behavior:
- fit/re-fit Isolation Forest independently on each incoming batch
- score current batch only (no adaptive memory beyond batch)
- stitch per-batch scores into full stream output
"""

from __future__ import annotations

import numpy as np

from streaming_orchestrator import LatencyTracker, StreamingOrchestrator


def run_iforest_batch_streaming(
    orchestrator: StreamingOrchestrator, tracker: LatencyTracker
) -> np.ndarray:
    raise NotImplementedError("run_iforest_batch_streaming is not implemented yet.")
