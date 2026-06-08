# Streamify — Time Series Anomaly Detection in Streaming Settings

An experimental study evaluating three classical anomaly detection methods — **Isolation Forest**, **OCSVM**, and **LSTM** — in both offline (batch) and streaming settings, benchmarked against the **SAND** streaming baseline. Each method is tested in its standard offline form and in two streaming variants: a naïve batch adaptation (V1) and an improved adaptation that addresses concept drift or catastrophic forgetting (V2). Experiments are conducted across **10 time series** spanning **5 normality levels** of increasing cross-domain complexity.

---

## Table of Contents

- [Methods](#methods)
- [Normality Levels](#normality-levels)
- [Datasets](#datasets)
- [Running the Notebook](#running-the-notebook)
- [Project Structure](#project-structure)
- [Outputs](#outputs)

---

## Methods

### Offline Baselines

| Method | Description |
|--------|-------------|
| **IForest** | Isolation Forest trained on the full time series using a sliding-window feature representation. |
| **OCSVM** | One-Class SVM with RBF kernel, trained on the first 10% of each series. |
| **LSTM** | LSTM autoencoder trained on clean subsequences; anomaly score based on reconstruction error via Fourier distance. |
| **SAND** | Online SAND — the streaming baseline from the TSB-UAD benchmark, included for direct comparison. |

### Streaming Variants

| Method | Variant 1 — Naïve | Variant 2 — Improved |
|--------|-------------------|----------------------|
| **IForest** | Processes each incoming batch independently; model retrained from scratch on every batch. | **Drift-Aware IForest**: trains on the first batch; uses EMA of anomaly scores to detect concept drift; only retrains when drift is detected (`drift_threshold = 2.0`). |
| **OCSVM** | Applies OCSVM to each batch independently with no state carried across batches. | **KMeans-Warm OCSVM**: trains on the first batch; subsequent batches are scored using the existing model, which is updated only when the cluster structure shifts significantly. |
| **LSTM** | Retrains the LSTM from scratch on each batch independently — no memory across batches. | **Experience Replay LSTM**: maintains a replay buffer of low-anomaly-score (normal) samples from past batches; each new batch is trained on current data + buffer samples to prevent catastrophic forgetting. |

---

## Normality Levels

The benchmark is organised into five normality levels that progressively increase cross-domain complexity. N1 uses individual single-domain series; N2–N5 are constructed by concatenating series from different domains to simulate distribution shifts in a streaming scenario.

| Level | Description | Streams |
|-------|-------------|---------|
| **N1** | Single-domain series, no distribution shift | Yahoo, Dodgers, MGAB, ECG |
| **N2** | Two-domain concatenation — cross-domain shift | KDD21 + Yahoo, ECG + MGAB |
| **N3** | Three-domain concatenation | GHL + MGAB + SensorScope, SMAP + MSL + SMD |
| **N4** | Four-domain concatenation | ECG + Occupancy + Dodgers + Yahoo |
| **N5** | Five-domain concatenation — maximum heterogeneity | MSL + SMAP + MGAB + Dodgers + SensorScope |

---

## Datasets

All raw data comes from the [TSB-UAD-Public](https://github.com/TheDatumOrg/TSB-UAD) benchmark. Download the archive and extract it under `data/raw/`.

```
https://www.thedatum.org/datasets/TSB-UAD-Public.zip
```

After extraction, the expected structure under `data/raw/` is:

```
data/raw/
├── ECG/                MBA_ECG805_data.out
├── YAHOO/              YahooA3Benchmark-TS42_data.out
├── Dodgers/            101-freeway-traffic.test.out
├── Occupancy/          room-occupancy.train.csv@5.out
├── NASA-SMAP/          T-3.test.out
├── NASA-MSL/           T-13.test.out
├── SMD/                machine-3-11.test.csv@35.out
├── MGAB/               1.test.out
├── KDD21/              148_UCR_Anomaly_Lab2Cmac011215EPG4_6000_17390_17520.out
├── GHL/                01_Lev_fault_Temp_corr_seed_11_vars_23.test.csv@4.out
└── SensorScope/        stb-2.test.out
```

Each `.out` file contains two columns: the time series values and a binary anomaly label.

---

## Running the Notebook

There are two ways to run the experiments — choose whichever fits your setup.

### Option 1 — Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KJ0Am3HKL1kJlk9lhpyoVbI43nJzOc_W?usp=sharing)

**1. Get access to the datasets on Google Drive**

The datasets and generated normality streams are already available in a shared Google Drive folder:

**[📁 Open shared Drive folder](https://drive.google.com/drive/folders/1ZFyQSgmssUhbzMHfEbDan7Fo4_2XHLaC?usp=sharing)**

Add a shortcut for TSB-UAD-Public folder to your Drive under `My Drive/Time-Series Anomaly Detection/` so the notebook can find them at the expected path. Alternatively, download [TSB-UAD-Public.zip](https://www.thedatum.org/datasets/TSB-UAD-Public.zip), extract it, and place the folders under:

```
My Drive/Time-Series Anomaly Detection/TSB-UAD-Public/
```

The expected structure matches the one described in [Datasets](#datasets) above.

**2. Open the notebook and click Run All**

The notebook handles the rest automatically:
- Clones TSB-UAD from GitHub and installs all dependencies
- Mounts your Google Drive (a one-time authorization popup will appear)
- Reads datasets from Drive and saves generated normality streams to `My Drive/Time-Series Anomaly Detection/Concat-Data/`

> **Alternative — no Drive required:** If you prefer not to use Google Drive, uncomment the following lines in the notebook instead of mounting Drive:
> ```python
> # !wget https://www.thedatum.org/datasets/TSB-UAD-Public.zip
> # !unzip TSB-UAD-Public.zip -d data/
> ```
> This downloads the datasets directly into the Colab runtime. Note that the data will need to be re-downloaded each time the session is restarted.

---

### Option 2 — Local (Jupyter / VS Code)

### Prerequisites

**Download and place the datasets** (see [Datasets](#datasets) above) — this is the only manual step required.

Everything else (installing dependencies, cloning TSB-UAD, creating output directories) is handled automatically by the first cells of the notebook.

### Run

Open `notebooks/Streamify_Time_Series_Anomaly_Detection.ipynb` in Jupyter or VS Code and click **Run All**.

The notebook is self-contained and auto-detects the project root regardless of whether it is launched from the project root or from inside `notebooks/`.

### Caching

Every heavy computation cell checks whether its output already exists on disk before running. Normality streams are cached in `data/normalities/` and evaluation results in `results/metrics/`. If the file exists, the computation is skipped. This means:

- Running the full notebook once populates the cache.
- Subsequent **Run All** executions complete in seconds for already-computed results.
- To force a recomputation, delete the corresponding file from `results/metrics/`.

The same caching logic applies to the normality streams in `data/normalities/`.

---

## Project Structure

```
.
├── data/
│   ├── raw/                   # TSB-UAD-Public source files (not tracked by git)
│   └── normalities/           # Concatenated multi-domain streams N2–N5 (auto-generated)
├── notebooks/
│   ├── Streamify_Time_Series_Anomaly_Detection.ipynb
│   └── external/              # TSB-UAD library clone (not tracked by git)
├── results/
│   ├── metrics/               # CSV evaluation results per method/variant
│   └── figures/               # Generated plots (PNG, 150 dpi)
├── requirements.txt
└── README.md
```

---

## Outputs

### Metrics (`results/metrics/`)

| File | Method |
|------|--------|
| `isolation_forest_metrics.csv` | IForest — offline |
| `isolation_forest_streaming_metrics_df.csv` | IForest — V1 Naïve Streaming |
| `isolation_forest_simple_improved_streaming_metrics_df.csv` | IForest — V2 Drift-Aware |
| `ocsvm_metrics.csv` | OCSVM — offline |
| `ocsvm_metrics_batch_3000_2.csv` | OCSVM — V1 Naïve Streaming |
| `ocsvm_metrics_batch_3000_v2_2.csv` | OCSVM — V2 KMeans-Warm |
| `lstm_metrics.csv` | LSTM — offline |
| `lstm_naive_metrics.csv` | LSTM — V1 Naïve Streaming |
| `lstm_incremental_metrics.csv` | LSTM — V2 Experience Replay |
| `sand_metrics.csv` | SAND — online streaming baseline |

Each CSV contains per-dataset rows with columns: `dataset`, `AUC_ROC`, `F`, `runtime`, and method-specific metadata.

### Figures (`results/figures/`)

| File | Content |
|------|---------|
| `eda_file_length.png` | Time series length per dataset |
| `eda_anomalies.png` | Number of anomalous points per dataset |
| `eda_anomaly_pct.png` | Anomaly percentage per dataset |
| `baseline_auc_roc.png` | AUC-ROC: IForest vs OCSVM vs SAND (offline) |
| `baseline_fscore.png` | F-Score: IForest vs OCSVM vs SAND (offline) |
| `baseline_runtime.png` | Runtime: IForest vs OCSVM vs SAND (offline) |
| `iforest_v1_v2_AUC_ROC.png` | IForest — Naïve vs Drift-Aware streaming |
| `iforest_v1_v2_F.png` | IForest — F-Score streaming comparison |
| `iforest_v1_v2_runtime.png` | IForest — Runtime streaming comparison |
| `ocsvm_v1_v2_auc_roc.png` | OCSVM — Naïve vs KMeans-Warm streaming |
| `ocsvm_v1_v2_fscore.png` | OCSVM — F-Score streaming comparison |
| `ocsvm_v1_v2_runtime.png` | OCSVM — Runtime streaming comparison |
| `lstm_full_AUC_ROC.png` | LSTM — Offline vs Naïve vs Experience Replay |
| `lstm_full_F.png` | LSTM — F-Score three-way comparison |
| `lstm_full_runtime.png` | LSTM — Runtime three-way comparison |
| `poster_grand_heatmap_auc_roc.png` | AUC-ROC heatmap: all methods × all datasets |
| `poster_grand_heatmap_fscore.png` | F-Score heatmap: all methods × all datasets |
| `poster_avg_performance.png` | Average AUC-ROC bar chart across all methods |
| `poster_streaming_improvement.png` | Offline vs V1 vs V2 per method family |
| `poster_runtime_analysis.png` | Runtime heatmap + accuracy–runtime scatter |
| `poster_ts_samples.png` | Example time series with anomaly regions |
