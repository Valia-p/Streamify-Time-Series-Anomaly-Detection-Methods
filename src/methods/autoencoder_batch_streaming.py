"""Variant 1: Naive batch-wise streaming Autoencoder.

Expected behavior:
- train/apply autoencoder per batch with no long-term adaptation
- produce reconstruction-error scores for each batch
- concatenate scores to full-length stream output
"""

from __future__ import annotations

import numpy as np

from streaming_orchestrator import LatencyTracker, StreamingOrchestrator


def run_autoencoder_batch_streaming(
    orchestrator: StreamingOrchestrator, tracker: LatencyTracker
) -> np.ndarray:
    raise NotImplementedError("run_autoencoder_batch_streaming is not implemented yet.")
