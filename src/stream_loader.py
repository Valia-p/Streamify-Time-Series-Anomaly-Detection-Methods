from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List

import numpy as np
import pandas as pd


@dataclass
class StreamData:
    values: np.ndarray
    labels: np.ndarray
    metadata: Dict
    evaluation_mask: np.ndarray
    shift_indices: List[int]
    stream_name: str
    normality_level: int


def _derive_normality_level(stream_csv_path: Path) -> int:
    parent_name = stream_csv_path.parent.name
    if not parent_name.startswith("Normality"):
        raise ValueError(f"Cannot derive normality level from folder: {parent_name}")
    return int(parent_name.replace("Normality", ""))


def load_stream(
    stream_csv_path: str | Path,
    metadata_json_path: str | Path,
    init_length: int = 5000,
    subsequence_length: int = 200,
    boundary_exclusion: int = 200,
) -> StreamData:
    stream_csv_path = Path(stream_csv_path)
    metadata_json_path = Path(metadata_json_path)

    df = pd.read_csv(stream_csv_path)
    if "value" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Stream file must contain 'value' and 'label': {stream_csv_path}")

    values = df["value"].to_numpy(dtype=np.float64, copy=True)
    labels = (df["label"].to_numpy(copy=True) > 0).astype(np.int8)

    metadata = json.loads(metadata_json_path.read_text(encoding="utf-8"))
    shift_indices = [int(x) for x in metadata.get("boundary_indices", [])]

    evaluation_mask = np.ones(len(values), dtype=bool)
    warmup_end = min(len(values), int(init_length + subsequence_length))
    evaluation_mask[:warmup_end] = False

    for shift in shift_indices:
        start = max(0, int(shift - boundary_exclusion))
        end = min(len(values), int(shift + boundary_exclusion))
        evaluation_mask[start:end] = False

    expected_warmup_masked = init_length + subsequence_length
    assert (~evaluation_mask[:expected_warmup_masked]).all(), "Warmup mask incorrect"

    return StreamData(
        values=values,
        labels=labels,
        metadata=metadata,
        evaluation_mask=evaluation_mask,
        shift_indices=shift_indices,
        stream_name=stream_csv_path.stem,
        normality_level=_derive_normality_level(stream_csv_path),
    )


def iter_streams(
    streams_root: str | Path = "data/streams_znorm",
    init_length: int = 5000,
    subsequence_length: int = 200,
    boundary_exclusion: int = 200,
) -> Iterator[StreamData]:
    streams_root = Path(streams_root)
    for normality_dir in sorted(streams_root.glob("Normality*")):
        if not normality_dir.is_dir():
            continue
        for csv_path in sorted(normality_dir.glob("*.csv")):
            metadata_path = csv_path.with_suffix(".json")
            if not metadata_path.exists():
                raise FileNotFoundError(f"Missing metadata json for stream: {csv_path}")
            yield load_stream(
                stream_csv_path=csv_path,
                metadata_json_path=metadata_path,
                init_length=init_length,
                subsequence_length=subsequence_length,
                boundary_exclusion=boundary_exclusion,
            )
