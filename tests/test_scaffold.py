from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from methods.dummy_methods import run_dummy_random_offline, run_dummy_random_streaming
from metrics import compute_all_metrics
from runner import run_all
from stream_loader import load_stream
from streaming_orchestrator import StreamingOrchestrator


def _normality1_ecg_paths() -> tuple[Path, Path]:
    csv_path = PROJECT_ROOT / "data" / "streams_znorm" / "Normality1" / "ECG.csv"
    json_path = PROJECT_ROOT / "data" / "streams_znorm" / "Normality1" / "ECG.json"
    return csv_path, json_path


def test_stream_loader_mask_counts():
    csv_path, json_path = _normality1_ecg_paths()
    assert csv_path.exists()
    assert json_path.exists()

    stream = load_stream(csv_path, json_path)
    assert len(stream.values) == 25000
    assert stream.evaluation_mask.shape == stream.values.shape
    assert stream.evaluation_mask.sum() == 19800


def test_orchestrator_batches():
    csv_path, json_path = _normality1_ecg_paths()
    stream = load_stream(csv_path, json_path)
    orchestrator = StreamingOrchestrator(stream, batch_size=1000)
    assert orchestrator.total_batches() == 25
    assert sum(1 for _ in orchestrator.iter_batches()) == 25


def test_compute_all_metrics_dummy_scores():
    csv_path, json_path = _normality1_ecg_paths()
    stream = load_stream(csv_path, json_path)
    rng = np.random.default_rng(42)
    scores = rng.random(len(stream.values))
    results = compute_all_metrics(scores, stream.labels, stream.evaluation_mask, sliding_window=200)

    for k in ["VUS_PR", "VUS_ROC", "AUC_PR", "AUC_ROC"]:
        assert k in results


def test_runner_end_to_end_dummy(tmp_path: Path):
    csv_path, json_path = _normality1_ecg_paths()
    stream = load_stream(csv_path, json_path)

    out = run_all(
        streams=[stream],
        methods_offline=[("dummy_random_offline", run_dummy_random_offline)],
        methods_streaming=[("dummy_random_streaming", run_dummy_random_streaming)],
        output_dir=tmp_path / "run",
        batch_size=1000,
        sliding_window=200,
        seed=42,
    )

    metrics_path = Path(out["metrics_path"])
    latency_path = Path(out["latency_path"])
    run_log_path = Path(out["run_log_path"])

    assert metrics_path.exists()
    assert latency_path.exists()
    assert run_log_path.exists()

    metrics_df = pd.read_csv(metrics_path)
    for c in ["stream_name", "normality_level", "method_name", "mode", "VUS_PR", "VUS_ROC", "R_AUC_PR"]:
        assert c in metrics_df.columns
