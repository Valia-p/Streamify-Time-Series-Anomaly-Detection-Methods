import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse the same period estimation logic from stream_generator.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from stream_generator import estimate_period_autocorr


def quartile_anomaly_ratios(labels: np.ndarray) -> List[float]:
    n = len(labels)
    edges = np.linspace(0, n, 5, dtype=int)
    ratios: List[float] = []
    for i in range(4):
        s, e = int(edges[i]), int(edges[i + 1])
        if e <= s:
            ratios.append(0.0)
        else:
            ratios.append(float(np.mean(labels[s:e] > 0)))
    return ratios


def build_window_starts(length: int, segment_length: int, step: int) -> List[int]:
    if length < segment_length:
        return []
    starts = list(range(0, length - segment_length + 1, step))
    last = length - segment_length
    if starts[-1] != last:
        starts.append(last)
    return starts


def best_segment_by_anomaly_ratio(
    labels: np.ndarray,
    segment_length: int,
    step: int,
    target_ratio: float,
    min_ratio: float,
    max_ratio: float,
) -> Optional[Dict]:
    starts = build_window_starts(len(labels), segment_length, step)
    best: Optional[Dict] = None

    for start in starts:
        end = start + segment_length
        seg_ratio = float(np.mean(labels[start:end] > 0))
        if not (min_ratio <= seg_ratio <= max_ratio):
            continue

        distance = abs(seg_ratio - target_ratio)
        seg_count = int(np.sum(labels[start:end] > 0))
        candidate = {
            "best_start": int(start),
            "best_end": int(end),
            "best_segment_anomaly_ratio": seg_ratio,
            "best_segment_anomaly_count": seg_count,
            "anomaly_count_in_segment": seg_count,
            "distance_to_target_ratio": float(distance),
        }

        if best is None:
            best = candidate
            continue

        if candidate["distance_to_target_ratio"] < best["distance_to_target_ratio"]:
            best = candidate
            continue
        if candidate["distance_to_target_ratio"] == best["distance_to_target_ratio"]:
            if candidate["best_segment_anomaly_count"] > best["best_segment_anomaly_count"]:
                best = candidate
                continue
            if (
                candidate["best_segment_anomaly_count"] == best["best_segment_anomaly_count"]
                and candidate["best_start"] < best["best_start"]
            ):
                best = candidate

    return best


def plot_top_candidate(
    file_path: Path,
    output_path: Path,
    values: np.ndarray,
    labels: np.ndarray,
    best_start: int,
    best_end: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 5))
    x = np.arange(len(values))
    ax.plot(x, values, color="tab:blue", linewidth=0.8, label="value")

    anomaly_idx = np.where(labels > 0)[0]
    if anomaly_idx.size > 0:
        ax.scatter(
            anomaly_idx,
            values[anomaly_idx],
            color="tab:red",
            s=10,
            alpha=0.8,
            label="anomaly label",
        )

    ax.axvspan(best_start, best_end, color="tab:orange", alpha=0.22, label="best segment")
    ax.set_title(f"MGAB Top Candidate: {file_path.name}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def scan_mgab(args: argparse.Namespace) -> Dict:
    mgab_root = Path(args.mgab_root)
    files = sorted(mgab_root.glob("*.test.out")) + sorted(mgab_root.glob("*.train.out"))
    files = sorted(set(files))

    all_files: List[Dict] = []
    candidates_passing: List[Dict] = []

    for file_path in files:
        df = pd.read_csv(file_path, header=None)
        if df.shape[1] < 2:
            continue

        values = df.iloc[:, 0].astype(float).to_numpy()
        labels = df.iloc[:, 1].astype(float).to_numpy()
        length = int(len(values))

        anomaly_mask = labels > 0
        anomaly_count_full = int(np.sum(anomaly_mask))
        anomaly_ratio_full = float(np.mean(anomaly_mask))
        anomaly_indices = np.where(anomaly_mask)[0]
        if anomaly_indices.size == 0:
            anomaly_indices_range: List[Optional[int]] = [None, None]
        else:
            anomaly_indices_range = [int(anomaly_indices.min()), int(anomaly_indices.max())]

        densities = quartile_anomaly_ratios(labels)

        period_raw, period_strength, _ = estimate_period_autocorr(
            values, min_lag=args.min_lag, max_lag=args.max_lag
        )
        period_estimated = int(max(period_raw, args.min_pattern_length)) if period_raw > 0 else -1

        row: Dict = {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "length": length,
            "anomaly_ratio_full": anomaly_ratio_full,
            "anomaly_count_full": anomaly_count_full,
            "anomaly_indices_range": anomaly_indices_range,
            "anomaly_density_per_quartile": densities,
            "period_estimated_raw": int(period_raw) if period_raw > 0 else -1,
            "period_estimated": period_estimated,
            "period_strength": None if np.isnan(period_strength) else float(period_strength),
        }

        full_ratio_ok = args.full_ratio_min <= anomaly_ratio_full <= args.full_ratio_max
        row["full_ratio_candidate"] = bool(full_ratio_ok)

        if full_ratio_ok and length >= args.segment_length:
            best_segment = best_segment_by_anomaly_ratio(
                labels=labels,
                segment_length=args.segment_length,
                step=args.segment_step,
                target_ratio=args.target_segment_ratio,
                min_ratio=args.segment_ratio_min,
                max_ratio=args.segment_ratio_max,
            )
            if best_segment is not None:
                row.update(best_segment)
                row["full_periods_in_best_segment"] = (
                    float(args.segment_length / period_estimated) if period_estimated > 0 else None
                )
                candidates_passing.append(dict(row))

        all_files.append(row)

    candidates_passing.sort(key=lambda r: abs(r["best_segment_anomaly_ratio"] - args.target_segment_ratio))
    recommended = candidates_passing[0] if candidates_passing else None

    return {
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "scan_config": {
            "mgab_root": str(mgab_root),
            "segment_length": args.segment_length,
            "segment_step": args.segment_step,
            "target_segment_ratio": args.target_segment_ratio,
            "segment_ratio_bounds": [args.segment_ratio_min, args.segment_ratio_max],
            "full_ratio_bounds": [args.full_ratio_min, args.full_ratio_max],
            "period_min_lag": args.min_lag,
            "period_max_lag": args.max_lag,
            "period_floor": args.min_pattern_length,
        },
        "all_files": all_files,
        "candidates_passing": candidates_passing,
        "recommended": recommended,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan MGAB files and pick best anomaly-rich segment.")
    parser.add_argument("--mgab-root", default="data/raw/TSB-UAD-Public/MGAB")
    parser.add_argument("--output-json", default="data/mgab_scan_results.json")
    parser.add_argument("--segment-length", type=int, default=25000)
    parser.add_argument("--segment-step", type=int, default=1000)
    parser.add_argument("--target-segment-ratio", type=float, default=0.03)
    parser.add_argument("--segment-ratio-min", type=float, default=0.01)
    parser.add_argument("--segment-ratio-max", type=float, default=0.15)
    parser.add_argument("--full-ratio-min", type=float, default=0.005)
    parser.add_argument("--full-ratio-max", type=float, default=0.15)
    parser.add_argument("--min-lag", type=int, default=10)
    parser.add_argument("--max-lag", type=int, default=2000)
    parser.add_argument("--min-pattern-length", type=int, default=50)
    parser.add_argument("--plot-path", default="results/figures/sanity/mgab_top_candidate.png")
    args = parser.parse_args()

    results = scan_mgab(args)
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    recommended = results["recommended"]
    if recommended is None:
        print("WARNING: No MGAB candidate passed the configured criteria.")
        return

    rec_path = Path(recommended["file_path"])
    rec_df = pd.read_csv(rec_path, header=None)
    rec_values = rec_df.iloc[:, 0].astype(float).to_numpy()
    rec_labels = rec_df.iloc[:, 1].astype(float).to_numpy()

    plot_top_candidate(
        file_path=rec_path,
        output_path=Path(args.plot_path),
        values=rec_values,
        labels=rec_labels,
        best_start=int(recommended["best_start"]),
        best_end=int(recommended["best_end"]),
    )

    print(f"recommended file path: {recommended['file_path']}")
    print(f"recommended best_start: {recommended['best_start']}")
    print(f"best_segment_anomaly_ratio: {recommended['best_segment_anomaly_ratio']:.6f}")
    print(f"anomaly_count_in_segment: {recommended['anomaly_count_in_segment']}")
    print(
        "period_estimated, period_strength: "
        f"{recommended['period_estimated']}, {recommended['period_strength']}"
    )


if __name__ == "__main__":
    main()
