# Quickest Way to Run

If you want the least manual setup, copy your existing `data/` and `checkpoints/` folders into this project folder, then run:

```bash
python run_kii.py
```

The script will check the processed data and checkpoint, run preprocessing if raw NetCDF files are present, and then run the Kuroshio Instability Index evaluation.

On Windows, you can also double-click:

```text
run_kii_windows.bat
```

On macOS/Linux, you can run:

```bash
./run_kii_mac_linux.sh
```

---

# Kuroshio Current Anomaly Detection + Kuroshio Instability Index

This project extends the original **Kuroshio Current Anomaly Detection via Convolutional Autoencoder** pipeline into a more physics-oriented diagnostic framework.

The original goal was binary comparison against JMA large-amplitude meander (LAM) periods. The new extension reframes localized reconstruction error as an unsupervised **Kuroshio Instability Index (KII)**: a daily index of how strongly the Kuroshio surface-current field deviates from the quiescent-period flow manifold learned by the autoencoder.

## Research Reframing

Instead of asking only:

> Can the model classify JMA-defined LAM days?

this version asks:

> Can localized reconstruction error identify persistent Kuroshio dynamical-instability episodes, especially during pre-LAM or transition-stage periods?

This is useful because mature LAM labels and localized flow instability are not identical. A partial mismatch with the JMA LAM catalog can therefore be interpreted as evidence that the model is detecting localized instability rather than simply reproducing a path-regime label.

## Main Additions

### 1. Kuroshio Instability Index

`evaluate.py` now computes the following localized reconstruction-error statistics inside the Kuroshio ROI:

| Index | Definition | Interpretation |
|---|---|---|
| `I_mean` | Mean ROI reconstruction error | Domain-averaged baseline |
| `I_top10` | Mean of top 10% error pixels | Broad localized anomaly |
| `I_top5` | Mean of top 5% error pixels | Recommended robust KII source |
| `I_top1` | Mean of top 1% error pixels | Stronger localized anomaly |
| `I_max` | Maximum ROI error | Most sensitive local signal |
| `I_area95` | Fraction of ROI pixels above validation pixel threshold | Spatial extent of high-error anomaly |

The default KII source is:

```bash
--kii_score I_top5
```

KII is standardized using validation-period statistics:

```text
KII(t) = (I_top5(t) - mean_validation(I_top5)) / std_validation(I_top5)
```

The threshold is calibrated from the smoothed validation-period KII percentile.

### 2. Persistent Instability Episode Detection

The script detects continuous periods where smoothed KII exceeds the validation-calibrated threshold for at least `--min_duration` days.

Output:

```text
results/kii/instability_episodes.csv
```

Each episode includes:

```text
start_date, end_date, duration_days, peak_date, peak_KII, mean_KII,
overlap_LAM_days, overlap_LAM_ratio,
overlap_pre_LAM_days, overlap_pre_LAM_ratio
```

### 3. Pre-LAM Window Analysis

The script summarizes KII behavior before documented LAM onset dates using 30-, 60-, and 90-day windows.

Output:

```text
results/kii/pre_lam_window_stats.csv
```

This supports cautious phrasing such as **pre-LAM intensification signal** or **transition-stage instability**, without claiming deterministic prediction.

### 4. Score-Method Comparison

The script compares `I_mean`, `I_top10`, `I_top5`, `I_top1`, `I_max`, and `I_area95` against the JMA LAM period as a weak external reference.

Outputs:

```text
results/kii/score_method_comparison.csv
results/kii/score_method_comparison.png
```

### 5. KII Figures

New plots:

```text
results/kii/kii_timeseries_2019_2020.png
results/kii/kii_zoom_lam_transition.png
results/heatmaps/kii_rank01_YYYY-MM-DD.png
```

## Project Structure

```text
kuroshio_kii_project/
├── download_data.py
├── preprocess.py
├── model.py
├── train.py
├── evaluate.py
├── instability_index.py
├── run_experiments.py
├── requirements.txt
├── README.md
└── README_original.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download CMEMS data

```bash
copernicusmarine login
python download_data.py
```

### 3. Preprocess

```bash
python preprocess.py
```

The updated preprocessing script saves:

```text
data/processed/kuroshio_frames.npy
data/processed/land_mask.npy
data/processed/dates.npy
data/processed/lat.npy
data/processed/lon.npy
data/processed/split_indices.npz
data/processed/norm_stats.npz
```

### 4. Train

```bash
python train.py --epochs 100 --batch_size 16 --device cuda
```

### 5. Evaluate with KII enabled

Recommended first run:

```bash
python evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --device cpu \
  --score_mode topk_mean \
  --topk_percent 5 \
  --percentile 90 \
  --kii_score I_top5 \
  --kii_percentile 95 \
  --smooth_window 7 \
  --min_duration 5
```

More sensitive version:

```bash
python evaluate.py \
  --checkpoint checkpoints/best_model.pt \
  --device cpu \
  --score_mode max \
  --percentile 85 \
  --kii_score I_max \
  --kii_percentile 95
```

### 6. Run score comparison experiments

```bash
python run_experiments.py
```

## Important Outputs

```text
results/anomaly_scores.csv
results/score_timeseries.png
results/heatmaps/kii_rank01_YYYY-MM-DD.png

results/kii/kii_daily_scores.csv
results/kii/kii_timeseries_2019_2020.png
results/kii/kii_zoom_lam_transition.png
results/kii/instability_episodes.csv
results/kii/pre_lam_window_stats.csv
results/kii/score_method_comparison.csv
results/kii/score_method_comparison.png
results/kii/kii_metadata.json
```

## Suggested Research Wording

A suitable description for a professor update or proposal revision:

> I am extending the project by reframing localized reconstruction error as an unsupervised Kuroshio Instability Index, rather than using it only as a binary classifier for JMA-defined large-amplitude meanders. The index is based on top-k reconstruction-error statistics, high-error area ratios, and persistent anomaly episodes. This reframing is motivated by the observation that localized reconstruction error may be more sensitive to transition-stage dynamical instability than to mature LAM states alone.

## Notes

- The package does not include CMEMS data or trained checkpoints.
- `README_original.md` is kept for reference.
- The KII extension is intentionally implemented as an analysis layer after reconstruction-error generation, so it can be debugged without changing the autoencoder architecture.
