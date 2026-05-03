from __future__ import annotations

import warnings
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from TSB_UAD.vus.metrics import get_metrics

# Common metric names returned by TSB-UAD.
BASE_METRIC_KEYS: List[str] = [
    "AUC_ROC",
    "AUC_PR",
    "Precision",
    "Recall",
    "F",
    "Precision_at_k",
    "Rprecision",
    "Rrecall",
    "RF",
    "R_AUC_ROC",
    "R_AUC_PR",
    "VUS_ROC",
    "VUS_PR",
    "Affiliation_Precision",
    "Affiliation_Recall",
]


def _nan_metric_payload(warning_text: str) -> Dict:
    payload = {k: float("nan") for k in BASE_METRIC_KEYS}
    payload["warning"] = warning_text
    return payload


def compute_all_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    sliding_window: int = 200,
) -> Dict:
    if scores.shape != labels.shape:
        raise ValueError("scores and labels must have same shape")
    if mask.shape != labels.shape:
        raise ValueError("mask must match shape")

    scores_masked = scores[mask].astype(float)
    labels_masked = labels[mask].astype(int)

    if scores_masked.size == 0:
        return _nan_metric_payload("empty_masked_region")
    if labels_masked.sum() == 0:
        return _nan_metric_payload("no_anomalies_in_mask")

    try:
        results = get_metrics(
            scores_masked,
            labels_masked,
            metric="all",
            slidingWindow=sliding_window,
        )
        out = dict(results)
        out.setdefault("warning", "")
        return out
    except Exception as exc:  # pragma: no cover - defensive wrapper path
        warnings.warn(f"metric computation failed: {exc}")
        return _nan_metric_payload(f"metrics_exception:{type(exc).__name__}")


def format_metrics_table(results_list: Iterable[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(list(results_list))
    if df.empty:
        return df

    leading_cols = ["stream_name", "normality_level", "method_name", "mode"]
    metric_cols = [c for c in df.columns if c not in leading_cols]
    ordered = [c for c in leading_cols if c in df.columns] + sorted(metric_cols)
    df = df[ordered]

    sort_cols = [c for c in leading_cols if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df
