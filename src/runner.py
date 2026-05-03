from __future__ import annotations

import json
import random
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from metrics import compute_all_metrics, format_metrics_table
from stream_loader import StreamData
from streaming_orchestrator import LatencyTracker, StreamingOrchestrator

OfflineRunner = Callable[[np.ndarray], np.ndarray]
StreamingRunner = Callable[[StreamingOrchestrator, LatencyTracker], np.ndarray]


def set_global_seeds(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def run_method_on_stream(
    method_name: str,
    method_runner: OfflineRunner | StreamingRunner,
    stream_data: StreamData,
    mode: str,
    batch_size: int = 1000,
    sliding_window: int = 200,
) -> Dict:
    if mode not in ("offline", "streaming"):
        raise ValueError("mode must be one of: offline, streaming")

    t0 = time.perf_counter()
    latency_tracker = LatencyTracker()

    if mode == "offline":
        scores = np.asarray(method_runner(stream_data.values), dtype=float)
        latency_stats = latency_tracker.summary()
    else:
        orchestrator = StreamingOrchestrator(stream_data, batch_size=batch_size)
        scores = np.asarray(method_runner(orchestrator, latency_tracker), dtype=float)
        latency_stats = latency_tracker.summary()

    runtime_total = float(time.perf_counter() - t0)
    if scores.shape != stream_data.labels.shape:
        raise ValueError(
            f"{method_name} produced invalid score length. "
            f"Expected {stream_data.labels.shape}, got {scores.shape}"
        )

    metric_results = compute_all_metrics(
        scores=scores,
        labels=stream_data.labels,
        mask=stream_data.evaluation_mask,
        sliding_window=sliding_window,
    )

    return {
        "scores": scores,
        "metrics": metric_results,
        "latency_stats": latency_stats,
        "runtime_total": runtime_total,
    }


def run_all(
    streams: Iterable[StreamData],
    methods_offline: Sequence[Tuple[str, OfflineRunner]],
    methods_streaming: Sequence[Tuple[str, StreamingRunner]],
    output_dir: str | Path,
    batch_size: int = 1000,
    sliding_window: int = 200,
    seed: int = 42,
) -> Dict:
    set_global_seeds(seed=seed)

    output_dir = Path(output_dir)
    scores_dir = output_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: List[Dict] = []
    latency_rows: List[Dict] = []
    run_log: List[Dict] = []

    stream_list = list(streams)

    for stream in stream_list:
        method_plan: List[Tuple[str, str, Callable]] = []
        method_plan.extend([("offline", name, fn) for name, fn in methods_offline])
        method_plan.extend([("streaming", name, fn) for name, fn in methods_streaming])

        for mode, method_name, method_runner in method_plan:
            try:
                result = run_method_on_stream(
                    method_name=method_name,
                    method_runner=method_runner,
                    stream_data=stream,
                    mode=mode,
                    batch_size=batch_size,
                    sliding_window=sliding_window,
                )

                score_file = scores_dir / f"{stream.stream_name}__{method_name}__{mode}.npy"
                np.save(score_file, result["scores"])

                metric_row = {
                    "stream_name": stream.stream_name,
                    "normality_level": stream.normality_level,
                    "method_name": method_name,
                    "mode": mode,
                    **result["metrics"],
                }
                metrics_rows.append(metric_row)

                latency_row = {
                    "stream_name": stream.stream_name,
                    "normality_level": stream.normality_level,
                    "method_name": method_name,
                    "mode": mode,
                    "runtime_total": result["runtime_total"],
                    **result["latency_stats"],
                }
                latency_rows.append(latency_row)

                run_log.append(
                    {
                        "stream_name": stream.stream_name,
                        "normality_level": stream.normality_level,
                        "method_name": method_name,
                        "mode": mode,
                        "status": "ok",
                        "score_file": str(score_file),
                    }
                )
            except Exception as exc:  # pragma: no cover - defensive path
                warnings.warn(f"run failed for {stream.stream_name}/{method_name}/{mode}: {exc}")
                run_log.append(
                    {
                        "stream_name": stream.stream_name,
                        "normality_level": stream.normality_level,
                        "method_name": method_name,
                        "mode": mode,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    metrics_df = format_metrics_table(metrics_rows)
    latency_df = pd.DataFrame(latency_rows)
    if not latency_df.empty:
        sort_cols = [c for c in ["stream_name", "normality_level", "method_name", "mode"] if c in latency_df.columns]
        latency_df = latency_df.sort_values(sort_cols).reset_index(drop=True)

    metrics_path = output_dir / "metrics.csv"
    latency_path = output_dir / "latency.csv"
    run_log_path = output_dir / "run_log.json"

    metrics_df.to_csv(metrics_path, index=False)
    latency_df.to_csv(latency_path, index=False)
    run_log_path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "seed": seed,
                "batch_size": batch_size,
                "sliding_window": sliding_window,
                "n_streams": len(stream_list),
                "n_offline_methods": len(methods_offline),
                "n_streaming_methods": len(methods_streaming),
                "runs": run_log,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "metrics_path": str(metrics_path),
        "latency_path": str(latency_path),
        "run_log_path": str(run_log_path),
        "metrics_df": metrics_df,
        "latency_df": latency_df,
    }
