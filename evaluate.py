from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from instability_index import (
    DEFAULT_LAM_PERIODS,
    add_kii_columns,
    classification_stats,
    compare_score_methods,
    compute_pixel_threshold,
    compute_score_table,
    detect_episodes,
    enforce_min_duration,
    make_period_mask,
    make_pre_lam_mask,
    moving_average,
    plot_kii_timeseries,
    plot_kii_zoom,
    plot_score_method_comparison,
    pre_lam_window_stats,
)
from model import KuroshioAutoencoder, pixel_error_map

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
HEATMAP_DIR = RESULTS_DIR / "heatmaps"
KII_DIR = RESULTS_DIR / "kii"

JMA_LAM_PERIODS = DEFAULT_LAM_PERIODS

FULL_LON_MIN, FULL_LON_MAX = 130.0, 145.0
FULL_LAT_MIN, FULL_LAT_MAX = 25.0, 40.0

# Default ROI for Kuroshio/LAM-oriented localized scoring.
ROI_LON_MIN, ROI_LON_MAX = 132.0, 140.0
ROI_LAT_MIN, ROI_LAT_MAX = 30.0, 35.0


def load_model(
    checkpoint_path: str,
    device: torch.device,
    args_override: dict | None = None,
) -> KuroshioAutoencoder:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("args", {})
    if args_override:
        cfg.update(args_override)

    model = KuroshioAutoencoder(
        in_channels=2,
        base_filters=cfg.get("base_filters", 32),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_dates(n_frames: int) -> pd.DatetimeIndex:
    dates_path = DATA_DIR / "dates.npy"
    if dates_path.exists():
        dates = np.load(dates_path, allow_pickle=True)
        if len(dates) == n_frames:
            return pd.to_datetime(dates.astype(str))
        print(
            f"[warn] {dates_path} length={len(dates)} does not match frames={n_frames}; "
            "falling back to daily dates from 2010-01-01."
        )
    base_date = pd.Timestamp("2010-01-01")
    return pd.DatetimeIndex([base_date + pd.Timedelta(days=i) for i in range(n_frames)])


def load_lon_lat(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    lon_path = DATA_DIR / "lon.npy"
    lat_path = DATA_DIR / "lat.npy"
    if lon_path.exists() and lat_path.exists():
        lon = np.load(lon_path, allow_pickle=True)
        lat = np.load(lat_path, allow_pickle=True)
        return lon, lat
    lon = np.linspace(FULL_LON_MIN, FULL_LON_MAX, width)
    lat = np.linspace(FULL_LAT_MIN, FULL_LAT_MAX, height)
    return lon, lat


def build_roi_mask(
    height: int,
    width: int,
    ocean_mask: np.ndarray,
    lon: np.ndarray | None = None,
    lat: np.ndarray | None = None,
    lon_min: float = ROI_LON_MIN,
    lon_max: float = ROI_LON_MAX,
    lat_min: float = ROI_LAT_MIN,
    lat_max: float = ROI_LAT_MAX,
) -> np.ndarray:
    """Build a ROI mask and combine it with the ocean mask."""
    if lon is None or lat is None:
        lon = np.linspace(FULL_LON_MIN, FULL_LON_MAX, width)
        lat = np.linspace(FULL_LAT_MIN, FULL_LAT_MAX, height)

    lon = np.asarray(lon)
    lat = np.asarray(lat)

    if lon.ndim == 1 and lat.ndim == 1:
        lon_mask = (lon >= lon_min) & (lon <= lon_max)
        lat_mask = (lat >= lat_min) & (lat <= lat_max)
        roi = np.outer(lat_mask, lon_mask)
    elif lon.shape == (height, width) and lat.shape == (height, width):
        roi = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
    else:
        print("[warn] Unsupported lon/lat shape for ROI mask; using linear extent fallback.")
        lon_fallback = np.linspace(FULL_LON_MIN, FULL_LON_MAX, width)
        lat_fallback = np.linspace(FULL_LAT_MIN, FULL_LAT_MAX, height)
        roi = np.outer(
            (lat_fallback >= lat_min) & (lat_fallback <= lat_max),
            (lon_fallback >= lon_min) & (lon_fallback <= lon_max),
        )

    roi = roi.astype(bool) & ocean_mask.astype(bool)
    if int(roi.sum()) == 0:
        raise ValueError("ROI mask contains zero ocean pixels. Check ROI bounds and land_mask.")
    return roi


def compute_frame_score(
    error_map: np.ndarray,
    score_mask: np.ndarray,
    score_mode: str = "topk_mean",
    topk_percent: float = 10.0,
) -> float:
    """
    Compute a scalar legacy anomaly score from one 2D error map.

    score_mode:
      - mean: mean error over ROI
      - topk_mean: mean of top-k percent ROI pixels
      - max: max error in ROI
    """
    roi_vals = error_map[score_mask.astype(bool)]
    roi_vals = roi_vals[np.isfinite(roi_vals)]

    if roi_vals.size == 0:
        return 0.0
    if score_mode == "mean":
        return float(np.mean(roi_vals))
    if score_mode == "max":
        return float(np.max(roi_vals))
    if score_mode == "topk_mean":
        topk_percent = float(np.clip(topk_percent, 0.01, 100.0))
        k = max(1, int(np.ceil(len(roi_vals) * topk_percent / 100.0)))
        top_vals = np.partition(roi_vals, -k)[-k:]
        return float(np.mean(top_vals))
    raise ValueError(f"Unsupported score_mode: {score_mode}")


def compute_legacy_scores(
    error_maps: np.ndarray,
    score_mask: np.ndarray,
    score_mode: str,
    topk_percent: float,
) -> np.ndarray:
    return np.asarray(
        [
            compute_frame_score(
                error_map=error_maps[i],
                score_mask=score_mask,
                score_mode=score_mode,
                topk_percent=topk_percent,
            )
            for i in range(error_maps.shape[0])
        ],
        dtype=np.float32,
    )


def compute_error_maps(
    model: KuroshioAutoencoder,
    data_tensor: torch.Tensor,
    batch_size: int = 8,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Run model inference and return pixel-wise reconstruction error maps."""
    ds = TensorDataset(data_tensor)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    maps_list = []

    with torch.no_grad():
        for (batch,) in dl:
            batch = batch.to(device)
            pred = model(batch)
            emap = pixel_error_map(pred, batch)
            maps_list.append(emap.cpu().numpy().astype(np.float32))

    return np.concatenate(maps_list, axis=0)


def calibrate_threshold(val_scores: np.ndarray, percentile: float = 90.0) -> float:
    thresh = float(np.percentile(val_scores[np.isfinite(val_scores)], percentile))
    print(f"Legacy threshold (validation {percentile:.0f}th pct): {thresh:.6f}")
    return thresh


def get_plot_extent(lon: np.ndarray, lat: np.ndarray) -> list[float]:
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    return [float(np.nanmin(lon)), float(np.nanmax(lon)), float(np.nanmin(lat)), float(np.nanmax(lat))]


def plot_heatmap(
    error_map: np.ndarray,
    display_mask: np.ndarray,
    date_str: str,
    pixel_threshold: float,
    frame_score: float,
    save_path: Path,
    lon: np.ndarray,
    lat: np.ndarray,
    roi_bounds: tuple[float, float, float, float],
) -> None:
    masked_map = np.where(display_mask.astype(bool), error_map, np.nan)
    extent = get_plot_extent(lon, lat)

    fig, ax = plt.subplots(figsize=(8, 6))
    vmax = np.nanpercentile(masked_map, 99)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = None
    im = ax.imshow(
        masked_map,
        origin="lower",
        extent=extent,
        cmap="hot_r",
        aspect="auto",
        vmin=0,
        vmax=vmax,
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Reconstruction error (normalized squared units)", fontsize=10)

    height, width = masked_map.shape
    x = np.linspace(extent[0], extent[1], width)
    y = np.linspace(extent[2], extent[3], height)

    if np.isfinite(pixel_threshold) and np.nanmax(masked_map) >= pixel_threshold:
        ax.contour(
            x,
            y,
            masked_map,
            levels=[pixel_threshold],
            colors="cyan",
            linewidths=1.1,
            linestyles="--",
        )

    lon_min, lon_max, lat_min, lat_max = roi_bounds
    ax.plot(
        [lon_min, lon_max, lon_max, lon_min, lon_min],
        [lat_min, lat_min, lat_max, lat_max, lat_min],
        color="lime",
        linewidth=1.2,
        linestyle="-",
    )

    ax.set_title(
        f"Kuroshio Reconstruction-Error Heatmap - {date_str}\n"
        f"Frame score = {frame_score:.4f}",
        fontsize=12,
    )
    ax.set_xlabel("Longitude (deg E)")
    ax.set_ylabel("Latitude (deg N)")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_legacy_timeseries(
    dates: pd.DatetimeIndex,
    raw_scores: np.ndarray,
    smooth_scores: np.ndarray,
    threshold: float,
    detected_flags: np.ndarray,
    save_path: Path,
    score_mode: str,
    topk_percent: float,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.plot(dates, raw_scores, color="#9ecae1", linewidth=0.8, alpha=0.7, label="Raw score")
    ax.plot(dates, smooth_scores, color="#1a6faf", linewidth=1.4, label="Smoothed score")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.2, label=f"Threshold = {threshold:.4f}")

    for start, end in JMA_LAM_PERIODS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="orange", alpha=0.22, label="JMA LAM period")

    flags = detected_flags.astype(bool)
    start_idx = None
    for i in range(len(flags) + 1):
        if i < len(flags) and flags[i]:
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None:
                ax.axvspan(dates[start_idx], dates[i - 1], color="green", alpha=0.15, label="Detected interval")
                start_idx = None

    title_suffix = score_mode
    if score_mode == "topk_mean":
        title_suffix += f" (top {topk_percent:.1f}%)"

    ax.set_title(
        "Kuroshio Current Legacy Anomaly Score - Test Period\n"
        f"ROI-based scoring with temporal smoothing | mode={title_suffix}",
        fontsize=12,
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Frame anomaly score")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen[label] = handle
    ax.legend(seen.values(), seen.keys(), loc="upper left", fontsize=9)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    print(f"Saved legacy time-series plot -> {save_path}")


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def evaluate(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    HEATMAP_DIR.mkdir(parents=True, exist_ok=True)
    KII_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device)

    frames = np.load(DATA_DIR / "kuroshio_frames.npy")
    ocean_mask = np.load(DATA_DIR / "land_mask.npy")
    splits = np.load(DATA_DIR / "split_indices.npz")

    n_frames, _, height, width = frames.shape
    all_dates = load_dates(n_frames)
    lon, lat = load_lon_lat(height, width)

    roi_bounds = (args.roi_lon_min, args.roi_lon_max, args.roi_lat_min, args.roi_lat_max)
    score_mask_np = build_roi_mask(
        height,
        width,
        ocean_mask,
        lon=lon,
        lat=lat,
        lon_min=args.roi_lon_min,
        lon_max=args.roi_lon_max,
        lat_min=args.roi_lat_min,
        lat_max=args.roi_lat_max,
    )

    val_idx = splits["val"]
    test_idx = splits["test"]
    x_val = torch.from_numpy(frames[val_idx])
    x_test = torch.from_numpy(frames[test_idx])
    val_dates = all_dates[val_idx]
    test_dates = all_dates[test_idx]

    print("Running validation inference...")
    val_error_maps = compute_error_maps(
        model=model,
        data_tensor=x_val,
        batch_size=args.batch_size,
        device=device,
    )
    print("Running test inference...")
    test_error_maps = compute_error_maps(
        model=model,
        data_tensor=x_test,
        batch_size=args.batch_size,
        device=device,
    )

    if args.save_error_maps:
        np.savez_compressed(
            RESULTS_DIR / "error_maps_test.npz",
            error_maps=test_error_maps,
            dates=test_dates.strftime("%Y-%m-%d").to_numpy(),
            score_mask=score_mask_np,
        )
        print(f"Saved compressed test error maps -> {RESULTS_DIR / 'error_maps_test.npz'}")

    pixel_threshold = compute_pixel_threshold(
        val_error_maps,
        score_mask_np,
        percentile=args.area_pixel_percentile,
    )
    print(f"Pixel threshold for high-error area ({args.area_pixel_percentile:.0f}th pct): {pixel_threshold:.6f}")

    val_score_df = compute_score_table(
        val_error_maps,
        dates=val_dates,
        score_mask=score_mask_np,
        pixel_threshold=pixel_threshold,
    )
    test_score_df = compute_score_table(
        test_error_maps,
        dates=test_dates,
        score_mask=score_mask_np,
        pixel_threshold=pixel_threshold,
    )

    # Legacy score path, kept for compatibility with the original project.
    val_scores_raw = compute_legacy_scores(
        val_error_maps,
        score_mask_np,
        score_mode=args.score_mode,
        topk_percent=args.topk_percent,
    )
    val_scores = moving_average(val_scores_raw, window=args.smooth_window)
    legacy_threshold = calibrate_threshold(val_scores, args.percentile)

    test_scores_raw = compute_legacy_scores(
        test_error_maps,
        score_mask_np,
        score_mode=args.score_mode,
        topk_percent=args.topk_percent,
    )
    test_scores = moving_average(test_scores_raw, window=args.smooth_window)
    detected_raw = (test_scores > legacy_threshold).astype(int)
    detected = enforce_min_duration(detected_raw, min_duration=args.min_duration)

    # KII path: standardize localized score using validation baseline.
    kii_df, kii_threshold, kii_meta = add_kii_columns(
        test_df=test_score_df,
        val_df=val_score_df,
        score_col=args.kii_score,
        smooth_window=args.smooth_window,
        threshold_percentile=args.kii_percentile,
    )
    kii_raw_flags = (kii_df["KII_smooth"].to_numpy() > kii_threshold).astype(int)
    kii_flags = enforce_min_duration(kii_raw_flags, min_duration=args.min_duration)
    kii_df["is_instability_raw"] = kii_raw_flags
    kii_df["is_instability"] = kii_flags

    lam_mask = make_period_mask(test_dates, JMA_LAM_PERIODS)
    pre_lam_mask = make_pre_lam_mask(test_dates, JMA_LAM_PERIODS, args.pre_lam_window)

    # Merge legacy and KII outputs into one daily CSV.
    out_df = kii_df.copy()
    out_df["date"] = pd.to_datetime(out_df["date"]).dt.strftime("%Y-%m-%d")
    out_df["raw_score"] = test_scores_raw
    out_df["smooth_score"] = test_scores
    out_df["threshold"] = legacy_threshold
    out_df["score_mode"] = args.score_mode
    out_df["topk_percent"] = args.topk_percent
    out_df["is_anomaly_raw"] = detected_raw
    out_df["is_anomaly"] = detected
    out_df["is_JMA_LAM"] = lam_mask.astype(int)
    out_df["is_pre_LAM_window"] = pre_lam_mask.astype(int)

    csv_path = RESULTS_DIR / "anomaly_scores.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"Saved daily anomaly/KII scores -> {csv_path}")

    kii_csv_path = KII_DIR / "kii_daily_scores.csv"
    out_df.to_csv(kii_csv_path, index=False)
    print(f"Saved KII daily scores -> {kii_csv_path}")

    episodes_df = detect_episodes(
        out_df,
        flag_col="is_instability",
        score_col="KII_smooth",
        lam_periods=JMA_LAM_PERIODS,
        pre_lam_window=args.pre_lam_window,
    )
    episodes_path = KII_DIR / "instability_episodes.csv"
    episodes_df.to_csv(episodes_path, index=False)
    print(f"Saved instability episodes -> {episodes_path}")

    pre_stats_df = pre_lam_window_stats(
        out_df,
        lam_periods=JMA_LAM_PERIODS,
        windows=args.pre_lam_windows,
        score_col="KII_smooth",
        threshold_col="KII_threshold",
    )
    pre_stats_path = KII_DIR / "pre_lam_window_stats.csv"
    pre_stats_df.to_csv(pre_stats_path, index=False)
    print(f"Saved pre-LAM window stats -> {pre_stats_path}")

    summary_df = compare_score_methods(
        val_score_df,
        test_score_df,
        lam_periods=JMA_LAM_PERIODS,
        smooth_window=args.smooth_window,
        threshold_percentile=args.kii_percentile,
        min_duration=args.min_duration,
    )
    summary_path = KII_DIR / "score_method_comparison.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved score-method comparison -> {summary_path}")

    metadata = {
        "project_reframing": "Localized reconstruction error is interpreted as an unsupervised Kuroshio Instability Index rather than only a binary LAM classifier.",
        "roi": {
            "lon_min": args.roi_lon_min,
            "lon_max": args.roi_lon_max,
            "lat_min": args.roi_lat_min,
            "lat_max": args.roi_lat_max,
            "ocean_pixels": int(score_mask_np.sum()),
        },
        "legacy_score": {
            "score_mode": args.score_mode,
            "topk_percent": args.topk_percent,
            "threshold_percentile": args.percentile,
            "threshold": legacy_threshold,
        },
        "kii": kii_meta,
        "pixel_threshold": {
            "percentile": args.area_pixel_percentile,
            "threshold": pixel_threshold,
        },
        "jma_lam_periods": JMA_LAM_PERIODS,
    }
    save_json(KII_DIR / "kii_metadata.json", metadata)

    # Heatmaps ranked by KII score, which is the new physics-oriented target.
    ranking_values = out_df["KII_smooth"].astype(float).to_numpy()
    top_indices = np.argsort(ranking_values)[-args.top_n:][::-1]
    for rank, idx in enumerate(top_indices, 1):
        date_str = test_dates[idx].strftime("%Y-%m-%d")
        save_path = HEATMAP_DIR / f"kii_rank{rank:02d}_{date_str}.png"
        plot_heatmap(
            error_map=test_error_maps[idx],
            display_mask=ocean_mask,
            date_str=date_str,
            pixel_threshold=pixel_threshold,
            frame_score=ranking_values[idx],
            save_path=save_path,
            lon=lon,
            lat=lat,
            roi_bounds=roi_bounds,
        )
        print(f"  KII heatmap rank {rank}: {date_str} KII={ranking_values[idx]:.3f} -> {save_path}")

    plot_legacy_timeseries(
        dates=test_dates,
        raw_scores=test_scores_raw,
        smooth_scores=test_scores,
        threshold=legacy_threshold,
        detected_flags=detected,
        save_path=RESULTS_DIR / "score_timeseries.png",
        score_mode=args.score_mode,
        topk_percent=args.topk_percent,
    )

    plot_kii_timeseries(
        out_df,
        save_path=KII_DIR / "kii_timeseries_2019_2020.png",
        lam_periods=JMA_LAM_PERIODS,
        title=f"Kuroshio Instability Index - {args.kii_score}",
    )
    plot_kii_zoom(
        out_df,
        save_path=KII_DIR / "kii_zoom_lam_transition.png",
        zoom_start=args.zoom_start,
        zoom_end=args.zoom_end,
        lam_periods=JMA_LAM_PERIODS,
    )
    plot_score_method_comparison(summary_df, KII_DIR / "score_method_comparison.png")

    print("\n-- Legacy Detection Statistics vs. JMA LAM ----------------")
    legacy_stats = classification_stats(detected.astype(bool), lam_mask)
    for key in ["TP", "FP", "FN", "TN", "precision", "recall", "f1", "accuracy"]:
        value = legacy_stats[key]
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")

    print("\n-- KII Instability Statistics vs. JMA LAM weak reference ---")
    kii_stats = classification_stats(kii_flags.astype(bool), lam_mask)
    for key in ["TP", "FP", "FN", "TN", "precision", "recall", "f1", "accuracy"]:
        value = kii_stats[key]
        print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")

    if len(episodes_df) > 0:
        print("\nTop instability episodes by peak KII:")
        print(episodes_df.sort_values("peak_KII", ascending=False).head(5).to_string(index=False))
    else:
        print("\nNo persistent KII episodes detected with the current threshold/min_duration.")

    print("\nEvaluation complete.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate Kuroshio Autoencoder and compute Kuroshio Instability Index"
    )

    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pt")
    p.add_argument("--percentile", type=float, default=90.0, help="Legacy score threshold percentile.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--smooth_window", type=int, default=7)
    p.add_argument("--min_duration", type=int, default=5)

    p.add_argument(
        "--score_mode",
        type=str,
        default="topk_mean",
        choices=["mean", "topk_mean", "max"],
        help="Legacy reduction from ROI error map to scalar frame score.",
    )
    p.add_argument("--topk_percent", type=float, default=10.0, help="Used only when score_mode=topk_mean.")
    p.add_argument("--top_n", type=int, default=5, help="Number of highest-KII heatmaps to save.")

    p.add_argument(
        "--kii_score",
        type=str,
        default="I_top5",
        choices=["I_mean", "I_top10", "I_top5", "I_top1", "I_max", "I_area95"],
        help="Localized score used as the Kuroshio Instability Index source.",
    )
    p.add_argument("--kii_percentile", type=float, default=95.0, help="Validation percentile for KII threshold.")
    p.add_argument(
        "--area_pixel_percentile",
        type=float,
        default=95.0,
        help="Validation pixel percentile used to define high-error area ratio.",
    )
    p.add_argument("--pre_lam_window", type=int, default=90)
    p.add_argument("--pre_lam_windows", type=int, nargs="+", default=[30, 60, 90])
    p.add_argument("--zoom_start", type=str, default="2019-05-01")
    p.add_argument("--zoom_end", type=str, default="2019-10-31")
    p.add_argument("--save_error_maps", action="store_true")

    p.add_argument("--roi_lon_min", type=float, default=ROI_LON_MIN)
    p.add_argument("--roi_lon_max", type=float, default=ROI_LON_MAX)
    p.add_argument("--roi_lat_min", type=float, default=ROI_LAT_MIN)
    p.add_argument("--roi_lat_max", type=float, default=ROI_LAT_MAX)

    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
