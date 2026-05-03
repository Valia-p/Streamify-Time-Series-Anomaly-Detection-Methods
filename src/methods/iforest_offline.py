"""Offline Isolation Forest baseline.

This module will host the full offline Isolation Forest implementation.
Expected behavior:
- build windowed features from the full stream
- fit Isolation Forest once on full input
- output aligned anomaly scores for the complete stream
"""

from __future__ import annotations

import numpy as np


def run_iforest_offline(values: np.ndarray) -> np.ndarray:
    raise NotImplementedError("run_iforest_offline is not implemented yet.")
