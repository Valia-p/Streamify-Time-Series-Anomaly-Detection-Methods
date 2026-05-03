"""Offline Autoencoder baseline.

This module will host the full offline autoencoder implementation.
Expected behavior:
- construct fixed-length windows from full stream
- train autoencoder offline on full input
- compute reconstruction-error scores and align back to timeline
"""

from __future__ import annotations

import numpy as np


def run_autoencoder_offline(values: np.ndarray) -> np.ndarray:
    raise NotImplementedError("run_autoencoder_offline is not implemented yet.")
