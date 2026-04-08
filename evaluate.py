import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from model import KuroshioAutoencoder, pixel_error_map

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
HEATMAP_DIR = RESULTS_DIR / "heatmaps"

JMA_LAM_PERIODS = [
    ("2019-08-01", "2020-09-30"),
]

FULL_LON_MIN, FULL_LON_MAX = 130.0, 145.0
FULL_LAT_MIN, FULL_LAT_MAX = 25.0, 40.0

# ROI for LAM-oriented scoring
ROI_LON_MIN, ROI_LON_MAX = 132.0, 140.0
ROI_LAT_MIN, ROI_LAT_MAX = 30.0, 35.0


def load_model(checkpoint_path: str, device: torch.device,
               args_override: dict = None) -> KuroshioAutoencoder:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("args", {})
    if args_override:
        cfg.update(args_override)

    model = KuroshioAutoencoder(
        in_channels=2,
        base_filters=cfg.get("base_filters", 32)
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def build_roi_mask(height: int, width: int, land_mask: np.ndarray) -> np.ndarray:
    lon = np.linspace(FULL_LON_MIN, FULL_LON_MAX, width)
    lat = np.linspace(FULL_LAT_MIN, FULL_LAT_MAX, height)

    lon_mask = (lon >= ROI_LON_MIN) & (lon <= ROI_LON_MAX)
    lat_mask = (lat >= ROI_LAT_MIN) & (lat <= ROI_LAT_MAX)

    roi = np.outer(lat_mask, lon_mask)
    roi = roi & land_mask.astype(bool)
    return roi.astype(np.float32)


def moving_average(x: np.ndarray, window: int = 7) -> np.ndarray:
    if window <= 1:
        return x.copy()
    s = pd.Series(x)
    return s.rolling(window=window, center=True, min_periods=1).mean().to_numpy()


def enforce_min_duration(flags: np.ndarray, min_duration: int = 5) -> np.ndarray:
    flags = flags.astype(bool).copy()
    n = len(flags)
    start = None

    for i in range(n + 1):
        if i < n and flags[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                length = i - start
                if length < min_duration:
                    flags[start:i] = False
                start = None

    return flags.astype(int)


def compute_scores(model, data_tensor: torch.Tensor,
                   score_mask: torch.Tensor,
                   batch_size: int = 8, device="cpu"):

    ds = TensorDataset(data_tensor)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    scores_list, maps_list = [], []

    with torch.no_grad():
        for (batch,) in dl:
            batch = batch.to(device)
            pred = model(batch)
            emap = pixel_error_map(pred, batch)

            emap_np = emap.cpu().numpy()
            masked = emap_np * score_mask.cpu().numpy()

            n_valid = score_mask.sum().item()
            score = masked.reshape(len(emap_np), -1).sum(axis=1) / max(n_valid, 1)

            scores_list.append(score)
            maps_list.append(emap_np)

    scores = np.concatenate(scores_list, axis=0)
    error_maps = np.concatenate(maps_list, axis=0)
    return scores, error_maps


def calibrate_threshold(val_scores: np.ndarray, percentile: float = 95) -> float:
    thresh = float(np.percentile(val_scores, percentile))
    print(f"Threshold (val {percentile:.0f}th pct): {thresh:.6f}")
    return thresh


def plot_heatmap(error_map: np.ndarray, display_mask: np.ndarray,
                 date_str: str, threshold: float,
                 save_path: Path) -> None:
    masked_map = np.where(display_mask, error_map, np.nan)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        masked_map,
        origin="lower",
        extent=[FULL_LON_MIN, FULL_LON_MAX, FULL_LAT_MIN, FULL_LAT_MAX],
        cmap="hot_r",
        aspect="auto",
        vmin=0,
        vmax=np.nanpercentile(masked_map, 99),
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Reconstruction Error  $(m/s)^2$", fontsize=10)

    h, w = masked_map.shape
    x = np.linspace(FULL_LON_MIN, FULL_LON_MAX, w)
    y = np.linspace(FULL_LAT_MIN, FULL_LAT_MAX, h)

    ax.contour(
        x, y, masked_map,
        levels=[threshold],
        colors="cyan",
        linewidths=1.2,
        linestyles="--"
    )

    # ROI box
    ax.plot(
        [ROI_LON_MIN, ROI_LON_MAX, ROI_LON_MAX, ROI_LON_MIN, ROI_LON_MIN],
        [ROI_LAT_MIN, ROI_LAT_MIN, ROI_LAT_MAX, ROI_LAT_MAX, ROI_LAT_MIN],
        color="lime",
        linewidth=1.2,
        linestyle="-"
    )

    ax.set_title(f"Kuroshio Anomaly Heatmap — {date_str}", fontsize=12)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_timeseries(dates: pd.DatetimeIndex,
                    raw_scores: np.ndarray,
                    smooth_scores: np.ndarray,
                    threshold: float,
                    detected_flags: np.ndarray,
                    save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(dates, raw_scores, color="#9ecae1", linewidth=0.8, alpha=0.7, label="Raw score")
    ax.plot(dates, smooth_scores, color="#1a6faf", linewidth=1.4, label="7-day moving average")
    ax.axhline(
        threshold, color="red", linestyle="--", linewidth=1.2,
        label=f"Threshold = {threshold:.4f}"
    )

    for (start, end) in JMA_LAM_PERIODS:
        ax.axvspan(
            pd.Timestamp(start), pd.Timestamp(end),
            color="orange", alpha=0.22, label="JMA LAM period"
        )

    detected_flags = detected_flags.astype(bool)
    start = None
    for i in range(len(detected_flags) + 1):
        if i < len(detected_flags) and detected_flags[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                ax.axvspan(
                    dates[start], dates[i - 1],
                    color="green", alpha=0.15, label="Detected interval"
                )
                start = None

    ax.set_xlabel("Date")
    ax.set_ylabel("Frame anomaly score  $(m/s)^2$")
    ax.set_title(
        "Kuroshio Current Anomaly Score — Test Period (2019–2020)\n"
        "ROI-based scoring with temporal smoothing",
        fontsize=12
    )
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), loc="upper left", fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved time-series plot → {save_path}")


def evaluate(args):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)

    frames = np.load(DATA_DIR / "kuroshio_frames.npy")
    land_mask = np.load(DATA_DIR / "land_mask.npy")
    splits = np.load(DATA_DIR / "split_indices.npz")

    h, w = land_mask.shape
    score_mask_np = build_roi_mask(h, w, land_mask)
    score_mask_t = torch.from_numpy(score_mask_np)

    val_idx = splits["val"]
    x_val = torch.from_numpy(frames[val_idx])
    val_scores_raw, _ = compute_scores(
        model, x_val, score_mask_t,
        batch_size=args.batch_size, device=device
    )
    val_scores = moving_average(val_scores_raw, window=args.smooth_window)
    threshold = calibrate_threshold(val_scores, args.percentile)

    test_idx = splits["test"]
    x_test = torch.from_numpy(frames[test_idx])
    test_scores_raw, error_maps = compute_scores(
        model, x_test, score_mask_t,
        batch_size=args.batch_size, device=device
    )
    test_scores = moving_average(test_scores_raw, window=args.smooth_window)

    base_date = pd.Timestamp("2010-01-01")
    test_dates = pd.DatetimeIndex([
        base_date + pd.Timedelta(days=int(idx)) for idx in test_idx
    ])

    detected_raw = (test_scores > threshold).astype(int)
    detected = enforce_min_duration(detected_raw, min_duration=args.min_duration)

    df = pd.DataFrame({
        "date": test_dates.strftime("%Y-%m-%d"),
        "raw_score": test_scores_raw,
        "smooth_score": test_scores,
        "is_anomaly_raw": detected_raw,
        "is_anomaly": detected,
    })
    csv_path = RESULTS_DIR / "anomaly_scores.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved anomaly scores → {csv_path}")

    top_indices = np.argsort(test_scores)[-5:][::-1]
    for rank, idx in enumerate(top_indices, 1):
        date_str = test_dates[idx].strftime("%Y-%m-%d")
        save_path = HEATMAP_DIR / f"rank{rank:02d}_{date_str}.png"
        plot_heatmap(error_maps[idx], land_mask, date_str, threshold, save_path)
        print(
            f"  Heatmap rank {rank}: {date_str}  "
            f"score={test_scores[idx]:.5f} → {save_path}"
        )

    plot_timeseries(
        test_dates,
        test_scores_raw,
        test_scores,
        threshold,
        detected,
        RESULTS_DIR / "score_timeseries.png"
    )

    print("\n── Detection Statistics vs. JMA LAM ─────────────────────────")
    lam_mask = np.zeros(len(test_dates), dtype=bool)
    for (start, end) in JMA_LAM_PERIODS:
        lam_mask |= (
            (test_dates >= pd.Timestamp(start)) &
            (test_dates <= pd.Timestamp(end))
        )

    detected_bool = detected.astype(bool)
    tp = (detected_bool & lam_mask).sum()
    fp = (detected_bool & ~lam_mask).sum()
    fn = (~detected_bool & lam_mask).sum()

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    print(f"  TP={tp}, FP={fp}, FN={fn}")
    print(f"  Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")
    print("\nEvaluation complete.")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Kuroshio Autoencoder")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    p.add_argument("--percentile", type=float, default=95.0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--smooth_window", type=int, default=7)
    p.add_argument("--min_duration", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())