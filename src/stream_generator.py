import argparse
import itertools
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import correlate, find_peaks


@dataclass
class SeriesConfig:
    name: str
    relative_path: str
    slice_start: int = 0


SELECTED_SERIES: List[SeriesConfig] = [
    SeriesConfig(name="ECG", relative_path="ECG/MBA_ECG14046_data_27.out", slice_start=0),
    SeriesConfig(
        name="OPPORTUNITY",
        relative_path="OPPORTUNITY/S1-ADL5.test.csv@128.out",
        slice_start=0,
    ),
    SeriesConfig(name="MGAB", relative_path="MGAB/4.test.out", slice_start=41000),
]


def estimate_period_autocorr(
    values: np.ndarray, min_lag: int, max_lag: int
) -> Tuple[int, float, np.ndarray]:
    """Estimate dominant period with normalized linear autocorrelation."""
    x = np.asarray(values, dtype=float)
    x = x - np.mean(x)
    std = np.std(x)
    if std == 0 or len(x) < (min_lag * 4):
        return -1, float("nan"), np.array([])
    x = x / std
    n = len(x)
    max_lag = min(max_lag, n // 2)
    if max_lag <= min_lag:
        return -1, float("nan"), np.array([])

    acf_full = correlate(x, x, mode="full", method="fft")
    acf = acf_full[n - 1 : n + max_lag + 1]
    acf = np.real(acf) / (acf[0] + 1e-12)

    segment = acf[min_lag : max_lag + 1]
    if segment.size == 0 or np.isnan(segment).all():
        return -1, float("nan"), acf

    peak_idxs, _ = find_peaks(segment, distance=max(5, min_lag // 2))
    if peak_idxs.size > 0:
        idx = int(peak_idxs[np.argmax(segment[peak_idxs])])
    else:
        idx = int(np.nanargmax(segment))
    lag = idx + min_lag
    peak_corr = float(segment[idx])
    return lag, peak_corr, acf


def estimate_period_fft(
    values: np.ndarray, min_period: int, max_period: int
) -> Tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    if n < 8:
        return float("nan"), float("nan")
    freqs = np.fft.rfftfreq(n, d=1.0)
    amps = np.abs(np.fft.rfft(x))
    if amps.size == 0:
        return float("nan"), float("nan")
    amps[0] = 0.0

    with np.errstate(divide="ignore", invalid="ignore"):
        periods = 1.0 / freqs
    valid = (
        np.isfinite(periods)
        & (periods >= float(min_period))
        & (periods <= float(max_period))
        & (amps > 0)
    )
    if not np.any(valid):
        return float("nan"), float("nan")

    valid_idx = np.where(valid)[0]
    best = int(valid_idx[np.argmax(amps[valid_idx])])
    return float(periods[best]), float(amps[best])


def select_period(
    acf_period: int,
    acf_strength: float,
    fft_period: float,
    min_pattern_length: int,
    consistency_factor: float,
) -> Tuple[int, str, bool, float]:
    ratio = float("nan")
    consistent = False

    has_acf = acf_period > 0 and not np.isnan(acf_strength)
    has_fft = not np.isnan(fft_period) and fft_period > 0

    if has_acf and has_fft:
        ratio = max(float(acf_period) / float(fft_period), float(fft_period) / float(acf_period))
        consistent = ratio <= consistency_factor
        if consistent:
            base = int(acf_period)
            source = "autocorr+fft"
        else:
            # Prefer autocorrelation if strong enough, otherwise defer to FFT.
            if acf_strength >= 0.10:
                base = int(acf_period)
                source = "autocorr_mismatch_fft"
            else:
                base = int(round(fft_period))
                source = "fft_mismatch_autocorr"
    elif has_acf:
        base = int(acf_period)
        source = "autocorr_only"
    elif has_fft:
        base = int(round(fft_period))
        source = "fft_only"
    else:
        return -1, "invalid", False, ratio

    if base < min_pattern_length:
        return int(min_pattern_length), f"{source}_floor", consistent, ratio
    return int(base), source, consistent, ratio


def z_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mean = float(np.mean(x))
    std = float(np.std(x))
    if std == 0:
        return np.zeros_like(x)
    return (x - mean) / std


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_stream(
    output_root: Path,
    level_name: str,
    stream_name: str,
    values: np.ndarray,
    labels: np.ndarray,
    metadata: dict,
) -> None:
    level_dir = output_root / level_name
    safe_mkdir(level_dir)

    csv_path = level_dir / f"{stream_name}.csv"
    meta_path = level_dir / f"{stream_name}.json"
    df = pd.DataFrame({"value": values.astype(float), "label": labels.astype(int)})
    df.to_csv(csv_path, index=False)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def build_stream_specs(keys: List[str]) -> Dict[str, List[Tuple[str, ...]]]:
    return {
        "Normality1": [(k,) for k in keys],
        "Normality2": list(itertools.permutations(keys, 2)),
        "Normality3": list(itertools.permutations(keys, 3)),
    }


def plot_sanity(
    fig_dir: Path,
    series_name: str,
    full_values: np.ndarray,
    segment_values: np.ndarray,
    segment_labels: np.ndarray,
    acf: np.ndarray,
    est_period: int,
    used_period: int,
    period_source: str,
    segment_length: int,
) -> None:
    safe_mkdir(fig_dir)
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    preview_len = min(6000, len(full_values))
    axes[0].plot(full_values[:preview_len], color="tab:blue", linewidth=0.8)
    axes[0].set_title(f"{series_name} - Raw Preview ({preview_len} points)")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Value")

    axes[1].plot(segment_values, color="tab:green", linewidth=0.8, label="segment")
    anomaly_idx = np.where(segment_labels > 0)[0]
    if anomaly_idx.size > 0:
        axes[1].scatter(
            anomaly_idx,
            segment_values[anomaly_idx],
            color="tab:red",
            s=8,
            alpha=0.7,
            label="anomaly label",
        )
    axes[1].set_title(
        f"{series_name} - Selected Segment ({segment_length} points, "
        f"anomaly ratio={np.mean(segment_labels > 0):.4f})"
    )
    axes[1].set_xlabel("Time in segment")
    axes[1].set_ylabel("Value")
    axes[1].legend(loc="upper right")

    if acf.size > 0:
        max_lag_plot = min(max(1000, used_period + 200), len(acf) - 1)
        lags = np.arange(1, max_lag_plot + 1)
        axes[2].plot(lags, acf[1 : max_lag_plot + 1], color="tab:purple", linewidth=1.0)
        if used_period > 0 and used_period <= max_lag_plot:
            axes[2].axvline(used_period, color="tab:red", linestyle="--", linewidth=1.2)
        axes[2].set_title(
            f"{series_name} - Autocorrelation (est={est_period}, used={used_period}, source={period_source})"
        )
    else:
        axes[2].text(
            0.5,
            0.5,
            "Autocorrelation not available",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )
        axes[2].set_title(f"{series_name} - Autocorrelation unavailable")
    axes[2].set_xlabel("Lag")
    axes[2].set_ylabel("ACF")

    periods_count = segment_length / max(1, used_period)
    fig.suptitle(
        f"{series_name} diagnostics | used_period={used_period} | full periods in segment={periods_count:.2f}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(fig_dir / f"{series_name}_sanity.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Normality1/2/3 stream datasets from selected TSB-UAD series."
    )
    parser.add_argument("--data-root", default="data/raw/TSB-UAD-Public")
    parser.add_argument("--output-raw", default="data/streams_raw")
    parser.add_argument("--output-znorm", default="data/streams_znorm")
    parser.add_argument("--summary-json", default="data/streams_summary.json")
    parser.add_argument("--figure-dir", default="results/figures/sanity")
    parser.add_argument("--segment-length", type=int, default=25000)
    parser.add_argument("--min-lag", type=int, default=10)
    parser.add_argument("--max-lag", type=int, default=2000)
    parser.add_argument("--period-lower-bound", type=int, default=10)
    parser.add_argument("--period-upper-bound", type=int, default=2000)
    parser.add_argument("--fallback-period", type=int, default=120)
    parser.add_argument("--min-pattern-length", type=int, default=50)
    parser.add_argument("--fft-consistency-factor", type=float, default=2.2)
    parser.add_argument("--min-period-repeats", type=int, default=20)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_raw = Path(args.output_raw)
    output_znorm = Path(args.output_znorm)
    figure_dir = Path(args.figure_dir)
    summary_json = Path(args.summary_json)

    safe_mkdir(output_raw)
    safe_mkdir(output_znorm)
    safe_mkdir(figure_dir)

    series_data = {}
    diagnostics = {}

    for cfg in SELECTED_SERIES:
        path = data_root / cfg.relative_path
        if not path.exists():
            raise FileNotFoundError(f"Missing source file: {path}")
        df = pd.read_csv(path, header=None)
        values = df.iloc[:, 0].astype(float).to_numpy()
        labels = df.iloc[:, 1].astype(int).to_numpy()

        est_period, strength, acf = estimate_period_autocorr(
            values, min_lag=args.min_lag, max_lag=args.max_lag
        )
        fft_period, fft_amp = estimate_period_fft(
            values, min_period=args.period_lower_bound, max_period=args.period_upper_bound
        )
        selected_period, period_source, period_consistent, period_ratio = select_period(
            acf_period=est_period,
            acf_strength=strength,
            fft_period=fft_period,
            min_pattern_length=args.min_pattern_length,
            consistency_factor=args.fft_consistency_factor,
        )
        use_fallback = (
            selected_period < args.period_lower_bound
            or selected_period > args.period_upper_bound
            or np.isnan(strength)
        )
        used_period = args.fallback_period if use_fallback else int(selected_period)
        if use_fallback:
            period_source = "fallback"

        series_data[cfg.name] = {
            "path": cfg.relative_path,
            "values": values,
            "labels": labels,
            "slice_start": int(cfg.slice_start),
        }
        diagnostics[cfg.name] = {
            "estimated_period_autocorr": int(est_period) if est_period >= 0 else -1,
            "estimated_period_fft": None if np.isnan(fft_period) else float(fft_period),
            "fft_amplitude": None if np.isnan(fft_amp) else float(fft_amp),
            "period_strength": float(strength) if not np.isnan(strength) else None,
            "autocorr_fft_ratio": None if np.isnan(period_ratio) else float(period_ratio),
            "autocorr_fft_consistent": bool(period_consistent),
            "selected_period_pre_bounds": int(selected_period) if selected_period >= 0 else -1,
            "used_period": int(used_period),
            "period_source": period_source,
            "length": int(len(values)),
            "slice_start": int(cfg.slice_start),
            "available_length_from_slice": int(len(values) - int(cfg.slice_start)),
            "anomaly_ratio_full": float(np.mean(labels > 0)),
            "acf": acf,
        }

    for key in diagnostics:
        if diagnostics[key]["available_length_from_slice"] <= 0:
            raise ValueError(
                f"Invalid slice_start for {key}: "
                f"{diagnostics[key]['slice_start']} with length {diagnostics[key]['length']}"
            )

    min_available_length = min(diagnostics[k]["available_length_from_slice"] for k in diagnostics)
    required_by_period = max(
        diagnostics[k]["used_period"] * args.min_period_repeats for k in diagnostics
    )
    final_segment_length = max(args.segment_length, required_by_period)
    if final_segment_length > min_available_length:
        final_segment_length = min_available_length

    # Pre-compute segments and save per-series diagnostics plots.
    selected_segments = {}
    for key in series_data:
        start = int(series_data[key]["slice_start"])
        end = start + final_segment_length
        segment_values = series_data[key]["values"][start:end]
        segment_labels = series_data[key]["labels"][start:end]
        selected_segments[key] = {
            "raw_values": segment_values,
            "znorm_values": z_normalize(segment_values),
            "labels": segment_labels,
        }
        diagnostics[key]["slice_end"] = int(end)
        diagnostics[key]["anomaly_ratio_segment"] = float(np.mean(segment_labels > 0))
        diagnostics[key]["full_periods_in_segment"] = float(
            final_segment_length / diagnostics[key]["used_period"]
        )
        plot_sanity(
            fig_dir=figure_dir,
            series_name=key,
            full_values=series_data[key]["values"],
            segment_values=segment_values,
            segment_labels=segment_labels,
            acf=diagnostics[key]["acf"],
            est_period=diagnostics[key]["estimated_period_autocorr"],
            used_period=diagnostics[key]["used_period"],
            period_source=diagnostics[key]["period_source"],
            segment_length=final_segment_length,
        )
        # ACF arrays are large and not needed in summary json.
        diagnostics[key].pop("acf", None)

    stream_specs = build_stream_specs(list(series_data.keys()))
    stream_count = {"Normality1": 0, "Normality2": 0, "Normality3": 0}

    for level_name, orders in stream_specs.items():
        for order in orders:
            stream_count[level_name] += 1
            stream_name = "__".join(order)
            raw_values = np.concatenate([selected_segments[k]["raw_values"] for k in order])
            znorm_values = np.concatenate(
                [selected_segments[k]["znorm_values"] for k in order]
            )
            labels = np.concatenate([selected_segments[k]["labels"] for k in order]).astype(
                int
            )

            local_periods = [diagnostics[k]["used_period"] for k in order]
            global_pattern_length = int(np.median(local_periods))
            recommended_subsequence_length = int(4 * global_pattern_length)
            boundary_indices = [
                i * final_segment_length for i in range(1, len(order))
            ]
            exclusion_half_window = recommended_subsequence_length
            boundary_exclusion_ranges = []
            for b in boundary_indices:
                boundary_exclusion_ranges.append(
                    {
                        "start": max(0, b - exclusion_half_window),
                        "end": min(len(labels), b + exclusion_half_window),
                    }
                )

            metadata = {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "stream_id": stream_name,
                "normality_level": int(level_name.replace("Normality", "")),
                "segment_order": list(order),
                "source_files": {k: series_data[k]["path"] for k in order},
                "source_slice_starts": {k: series_data[k]["slice_start"] for k in order},
                "normalization_note": (
                    "Per-segment z-normalization is stored in streams_znorm. "
                    "Raw values are stored in streams_raw."
                ),
                "segment_length": int(final_segment_length),
                "total_length": int(len(labels)),
                "global_pattern_length": global_pattern_length,
                "local_pattern_lengths": {k: diagnostics[k]["used_period"] for k in order},
                "recommended_subsequence_length": recommended_subsequence_length,
                "boundary_indices": boundary_indices,
                "boundary_exclusion_half_window": exclusion_half_window,
                "boundary_exclusion_ranges": boundary_exclusion_ranges,
                "boundary_handling": (
                    "Keep boundaries as natural transitions, and ignore "
                    "+/- recommended_subsequence_length around each boundary "
                    "during evaluation."
                ),
                "anomaly_ratio": float(np.mean(labels > 0)),
            }

            write_stream(
                output_root=output_raw,
                level_name=level_name,
                stream_name=stream_name,
                values=raw_values,
                labels=labels,
                metadata=metadata,
            )
            write_stream(
                output_root=output_znorm,
                level_name=level_name,
                stream_name=stream_name,
                values=znorm_values,
                labels=labels,
                metadata=metadata,
            )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selected_series": {
            k: {
                "path": series_data[k]["path"],
                "slice_start": series_data[k]["slice_start"],
                **diagnostics[k],
            }
            for k in series_data
        },
        "final_segment_length": int(final_segment_length),
        "stream_counts": stream_count,
        "outputs": {
            "raw_root": str(output_raw),
            "znorm_root": str(output_znorm),
            "sanity_figures": str(figure_dir),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Done.")
    print(f"segment_length={final_segment_length}")
    print(f"streams={stream_count}")
    print(f"summary={summary_json}")


if __name__ == "__main__":
    main()
