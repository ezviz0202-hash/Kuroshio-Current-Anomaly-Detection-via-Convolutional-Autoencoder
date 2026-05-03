"""
Utilities for the physics-oriented Kuroshio Instability Index (KII).

The core idea is to reinterpret localized reconstruction error from a convolutional
autoencoder as an unsupervised index of Kuroshio dynamical instability, rather
than using it only as a binary classifier for JMA-defined large-amplitude
meanders.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_LAM_PERIODS = [("2019-08-01", "2020-09-30")]


def moving_average(x: np.ndarray, window: int = 7) -> np.ndarray:
    """Centered rolling mean with edge handling."""
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x.copy()
    return (
        pd.Series(x)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def enforce_min_duration(flags: np.ndarray, min_duration: int = 5) -> np.ndarray:
    """Remove True intervals shorter than min_duration samples."""
    flags = np.asarray(flags).astype(bool).copy()
    if min_duration <= 1:
        return flags.astype(int)

    n = len(flags)
    start = None
    for i in range(n + 1):
        if i < n and flags[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                if i - start < min_duration:
                    flags[start:i] = False
                start = None
    return flags.astype(int)


def _valid_values(error_map: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is not None:
        vals = error_map[np.asarray(mask).astype(bool)]
    else:
        vals = error_map.reshape(-1)
    vals = np.asarray(vals, dtype=float)
    return vals[np.isfinite(vals)]


def topk_mean_from_values(values: np.ndarray, top_percent: float) -> float:
    """Mean of the largest top_percent percent values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    top_percent = float(np.clip(top_percent, 0.01, 100.0))
    k = max(1, int(np.ceil(values.size * top_percent / 100.0)))
    top_vals = np.partition(values, -k)[-k:]
    return float(np.mean(top_vals))


def compute_pixel_threshold(
    error_maps: np.ndarray,
    mask: np.ndarray,
    percentile: float = 95.0,
) -> float:
    """Pixel-level threshold from validation error maps within the scoring ROI."""
    mask = np.asarray(mask).astype(bool)
    vals = error_maps[:, mask].reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("No valid ROI values found for pixel threshold calibration.")
    return float(np.percentile(vals, percentile))


def compute_score_table(
    error_maps: np.ndarray,
    dates: Iterable,
    score_mask: np.ndarray,
    pixel_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Compute daily localized reconstruction-error statistics.

    Columns:
      I_mean   : mean reconstruction error in ROI
      I_top10  : mean of top 10 percent ROI pixels
      I_top5   : mean of top 5 percent ROI pixels
      I_top1   : mean of top 1 percent ROI pixels
      I_max    : maximum ROI error
      I_area95 : fraction of ROI pixels above a validation-calibrated pixel threshold
    """
    error_maps = np.asarray(error_maps)
    if error_maps.ndim != 3:
        raise ValueError(f"Expected error_maps with shape (time, H, W), got {error_maps.shape}")

    dates = pd.to_datetime(list(dates))
    if len(dates) != error_maps.shape[0]:
        raise ValueError("dates length does not match number of error maps")

    rows = []
    for i, date in enumerate(dates):
        vals = _valid_values(error_maps[i], score_mask)
        if vals.size == 0:
            row = {
                "date": date,
                "I_mean": np.nan,
                "I_top10": np.nan,
                "I_top5": np.nan,
                "I_top1": np.nan,
                "I_max": np.nan,
                "I_area95": np.nan,
            }
        else:
            row = {
                "date": date,
                "I_mean": float(np.mean(vals)),
                "I_top10": topk_mean_from_values(vals, 10.0),
                "I_top5": topk_mean_from_values(vals, 5.0),
                "I_top1": topk_mean_from_values(vals, 1.0),
                "I_max": float(np.max(vals)),
                "I_area95": (
                    float(np.mean(vals > pixel_threshold))
                    if pixel_threshold is not None
                    else np.nan
                ),
            }
        rows.append(row)

    return pd.DataFrame(rows)


def score_column_from_mode(score_mode: str, topk_percent: float) -> str | None:
    """Map legacy CLI score options to a standard KII score-table column."""
    if score_mode == "mean":
        return "I_mean"
    if score_mode == "max":
        return "I_max"
    if score_mode == "topk_mean":
        pct = float(topk_percent)
        if np.isclose(pct, 1.0):
            return "I_top1"
        if np.isclose(pct, 5.0):
            return "I_top5"
        if np.isclose(pct, 10.0):
            return "I_top10"
    return None


def generic_score_from_table_row(row: pd.Series, score_mode: str, topk_percent: float) -> float:
    """Use precomputed columns when possible."""
    col = score_column_from_mode(score_mode, topk_percent)
    if col is not None:
        return float(row[col])
    raise ValueError(
        "This generic lookup only supports mean, max, and top-k of 1/5/10 percent. "
        "Use compute_frame_score in evaluate.py for arbitrary top-k values."
    )


def add_kii_columns(
    test_df: pd.DataFrame,
    val_df: pd.DataFrame,
    score_col: str = "I_top5",
    smooth_window: int = 7,
    threshold_percentile: float = 95.0,
) -> tuple[pd.DataFrame, float, dict]:
    """
    Standardize a score column using validation-period statistics and add KII columns.

    Returns updated test_df, KII threshold, and metadata.
    """
    if score_col not in test_df.columns or score_col not in val_df.columns:
        raise ValueError(f"score_col={score_col!r} not found in score tables")

    val_scores = val_df[score_col].astype(float).to_numpy()
    val_scores = val_scores[np.isfinite(val_scores)]
    if val_scores.size == 0:
        raise ValueError(f"No finite validation scores for {score_col}")

    mu = float(np.mean(val_scores))
    sigma = float(np.std(val_scores))
    if sigma < 1e-12:
        sigma = 1.0

    val_kii_raw = (val_df[score_col].astype(float).to_numpy() - mu) / sigma
    val_kii_smooth = moving_average(val_kii_raw, smooth_window)
    threshold = float(np.percentile(val_kii_smooth[np.isfinite(val_kii_smooth)], threshold_percentile))

    out = test_df.copy()
    out["KII_source"] = score_col
    out["KII_raw"] = (out[score_col].astype(float) - mu) / sigma
    out[f"KII_{smooth_window}d"] = moving_average(out["KII_raw"].to_numpy(), smooth_window)
    out["KII_smooth"] = out[f"KII_{smooth_window}d"]
    out["KII_threshold"] = threshold

    meta = {
        "score_col": score_col,
        "validation_mean": mu,
        "validation_std": sigma,
        "smooth_window": smooth_window,
        "threshold_percentile": threshold_percentile,
        "threshold": threshold,
    }
    return out, threshold, meta


def make_period_mask(dates: Iterable, periods: Iterable[tuple[str, str]]) -> np.ndarray:
    dates = pd.to_datetime(list(dates))
    mask = np.zeros(len(dates), dtype=bool)
    for start, end in periods:
        mask |= (dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))
    return mask


def make_pre_lam_mask(
    dates: Iterable,
    periods: Iterable[tuple[str, str]],
    window_days: int = 90,
) -> np.ndarray:
    dates = pd.to_datetime(list(dates))
    mask = np.zeros(len(dates), dtype=bool)
    for start, _ in periods:
        start_ts = pd.Timestamp(start)
        mask |= (dates >= start_ts - pd.Timedelta(days=window_days)) & (dates < start_ts)
    return mask


def classification_stats(flags: np.ndarray, labels: np.ndarray) -> dict:
    flags = np.asarray(flags).astype(bool)
    labels = np.asarray(labels).astype(bool)
    tp = int((flags & labels).sum())
    fp = int((flags & ~labels).sum())
    fn = int((~flags & labels).sum())
    tn = int((~flags & ~labels).sum())
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def detect_episodes(
    df: pd.DataFrame,
    flag_col: str = "is_instability",
    score_col: str = "KII_smooth",
    lam_periods: Iterable[tuple[str, str]] = DEFAULT_LAM_PERIODS,
    pre_lam_window: int = 90,
) -> pd.DataFrame:
    """Convert daily instability flags into persistent episodes."""
    if len(df) == 0:
        return pd.DataFrame()

    local = df.copy()
    local["date"] = pd.to_datetime(local["date"])
    flags = local[flag_col].astype(bool).to_numpy()
    lam = make_period_mask(local["date"], lam_periods)
    pre = make_pre_lam_mask(local["date"], lam_periods, pre_lam_window)

    episodes = []
    start_idx = None
    for i in range(len(flags) + 1):
        if i < len(flags) and flags[i]:
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None:
                end_idx = i - 1
                segment = local.iloc[start_idx:end_idx + 1]
                seg_scores = segment[score_col].astype(float)
                peak_label = seg_scores.idxmax()
                peak_pos = local.index.get_loc(peak_label)
                seg_lam = lam[start_idx:end_idx + 1]
                seg_pre = pre[start_idx:end_idx + 1]
                duration = len(segment)
                episodes.append(
                    {
                        "episode_id": len(episodes) + 1,
                        "start_date": segment["date"].iloc[0].strftime("%Y-%m-%d"),
                        "end_date": segment["date"].iloc[-1].strftime("%Y-%m-%d"),
                        "duration_days": int(duration),
                        "peak_date": local["date"].iloc[peak_pos].strftime("%Y-%m-%d"),
                        "peak_KII": float(local[score_col].iloc[peak_pos]),
                        "mean_KII": float(seg_scores.mean()),
                        "overlap_LAM_days": int(seg_lam.sum()),
                        "overlap_LAM_ratio": float(seg_lam.sum() / max(duration, 1)),
                        "overlap_pre_LAM_days": int(seg_pre.sum()),
                        "overlap_pre_LAM_ratio": float(seg_pre.sum() / max(duration, 1)),
                    }
                )
                start_idx = None

    return pd.DataFrame(episodes)


def pre_lam_window_stats(
    df: pd.DataFrame,
    lam_periods: Iterable[tuple[str, str]] = DEFAULT_LAM_PERIODS,
    windows: Iterable[int] = (30, 60, 90),
    score_col: str = "KII_smooth",
    threshold_col: str = "KII_threshold",
) -> pd.DataFrame:
    """Summarize KII behavior before each LAM onset."""
    local = df.copy()
    local["date"] = pd.to_datetime(local["date"])
    threshold = float(local[threshold_col].iloc[0]) if threshold_col in local else np.nan
    rows = []
    for start, _ in lam_periods:
        start_ts = pd.Timestamp(start)
        for window in windows:
            mask = (local["date"] >= start_ts - pd.Timedelta(days=window)) & (local["date"] < start_ts)
            segment = local.loc[mask]
            if len(segment) == 0:
                continue
            rows.append(
                {
                    "lam_onset": start_ts.strftime("%Y-%m-%d"),
                    "window_days": int(window),
                    "window_start": (start_ts - pd.Timedelta(days=window)).strftime("%Y-%m-%d"),
                    "window_end": (start_ts - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    "mean_KII": float(segment[score_col].mean()),
                    "max_KII": float(segment[score_col].max()),
                    "days_above_threshold": int((segment[score_col] > threshold).sum()) if np.isfinite(threshold) else 0,
                    "fraction_above_threshold": float((segment[score_col] > threshold).mean()) if np.isfinite(threshold) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def compare_score_methods(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    lam_periods: Iterable[tuple[str, str]] = DEFAULT_LAM_PERIODS,
    score_cols: Iterable[str] = ("I_mean", "I_top10", "I_top5", "I_top1", "I_max", "I_area95"),
    smooth_window: int = 7,
    threshold_percentile: float = 95.0,
    min_duration: int = 5,
) -> pd.DataFrame:
    """Evaluate several localized score definitions against LAM as a weak reference."""
    lam = make_period_mask(test_df["date"], lam_periods)
    rows = []
    for col in score_cols:
        if col not in val_df.columns or col not in test_df.columns:
            continue
        if not np.isfinite(val_df[col].astype(float)).any():
            continue
        tmp, threshold, meta = add_kii_columns(
            test_df=test_df,
            val_df=val_df,
            score_col=col,
            smooth_window=smooth_window,
            threshold_percentile=threshold_percentile,
        )
        flags = enforce_min_duration((tmp["KII_smooth"] > threshold).to_numpy(), min_duration)
        stats = classification_stats(flags, lam)
        rows.append(
            {
                "score_col": col,
                "validation_mean": meta["validation_mean"],
                "validation_std": meta["validation_std"],
                "threshold": threshold,
                "smooth_window": smooth_window,
                "threshold_percentile": threshold_percentile,
                "min_duration": min_duration,
                **stats,
            }
        )
    return pd.DataFrame(rows)


def _dedupe_legend(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen[label] = handle
    if seen:
        ax.legend(seen.values(), seen.keys(), loc="upper left", fontsize=9)


def plot_kii_timeseries(
    df: pd.DataFrame,
    save_path: Path,
    lam_periods: Iterable[tuple[str, str]] = DEFAULT_LAM_PERIODS,
    title: str = "Kuroshio Instability Index",
    score_col: str = "KII_smooth",
    threshold_col: str = "KII_threshold",
) -> None:
    local = df.copy()
    local["date"] = pd.to_datetime(local["date"])
    threshold = float(local[threshold_col].iloc[0]) if threshold_col in local else np.nan

    fig, ax = plt.subplots(figsize=(14, 4.5))
    if "KII_raw" in local.columns:
        ax.plot(local["date"], local["KII_raw"], color="#bdbdbd", linewidth=0.8, alpha=0.75, label="Daily KII")
    ax.plot(local["date"], local[score_col], color="#08519c", linewidth=1.6, label="Smoothed KII")

    if np.isfinite(threshold):
        ax.axhline(threshold, color="#de2d26", linestyle="--", linewidth=1.2, label=f"Threshold = {threshold:.2f}")

    for start, end in lam_periods:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color="#fdae6b", alpha=0.25, label="JMA LAM period")

    if "is_instability" in local.columns:
        flags = local["is_instability"].astype(bool).to_numpy()
        start_idx = None
        for i in range(len(flags) + 1):
            if i < len(flags) and flags[i]:
                if start_idx is None:
                    start_idx = i
            else:
                if start_idx is not None:
                    ax.axvspan(
                        local["date"].iloc[start_idx],
                        local["date"].iloc[i - 1],
                        color="#74c476",
                        alpha=0.18,
                        label="Detected instability episode",
                    )
                    start_idx = None

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("KII z-score")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    _dedupe_legend(ax)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_kii_zoom(
    df: pd.DataFrame,
    save_path: Path,
    zoom_start: str = "2019-05-01",
    zoom_end: str = "2019-10-31",
    lam_periods: Iterable[tuple[str, str]] = DEFAULT_LAM_PERIODS,
) -> None:
    local = df.copy()
    local["date"] = pd.to_datetime(local["date"])
    mask = (local["date"] >= pd.Timestamp(zoom_start)) & (local["date"] <= pd.Timestamp(zoom_end))
    zoom = local.loc[mask].copy()
    if len(zoom) == 0:
        return
    plot_kii_timeseries(
        zoom,
        save_path=save_path,
        lam_periods=lam_periods,
        title=f"KII Zoom around LAM Transition ({zoom_start} to {zoom_end})",
    )


def plot_score_method_comparison(summary_df: pd.DataFrame, save_path: Path) -> None:
    if len(summary_df) == 0 or "f1" not in summary_df.columns:
        return
    data = summary_df.copy().sort_values("f1", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(data["score_col"], data["f1"])
    ax.set_ylim(0, max(1.0, float(data["f1"].max()) * 1.15))
    ax.set_ylabel("F1 vs JMA LAM reference")
    ax.set_xlabel("Localized reconstruction-error score")
    ax.set_title("Localized Scoring Methods for Kuroshio Instability")
    for idx, value in enumerate(data["f1"]):
        ax.text(idx, value + 0.015, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
