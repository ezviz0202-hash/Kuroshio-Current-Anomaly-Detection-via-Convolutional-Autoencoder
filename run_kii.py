"""One-command runner for the Kuroshio Instability Index extension.

Usage:
    python run_kii.py

This script does not include data or trained checkpoints. It will:
1. check whether data/processed exists;
2. run preprocess.py automatically if raw NetCDF files exist;
3. check whether checkpoints/best_model.pt exists;
4. optionally train a model if --train-if-missing is supplied;
5. run evaluate.py with KII outputs enabled.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n[run] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def has_processed_data() -> bool:
    required = [
        Path("data/processed/kuroshio_frames.npy"),
        Path("data/processed/land_mask.npy"),
        Path("data/processed/split_indices.npz"),
    ]
    return all(p.exists() for p in required)


def has_raw_data() -> bool:
    return Path("data/raw").exists() and any(Path("data/raw").glob("kuroshio_uv_*.nc"))


def has_checkpoint(path: str) -> bool:
    return Path(path).exists()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Kuroshio KII pipeline with minimal commands.")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--score-mode", default="topk_mean", choices=["mean", "topk_mean", "max"])
    parser.add_argument("--topk-percent", type=float, default=5.0)
    parser.add_argument("--percentile", type=float, default=85.0)
    parser.add_argument("--kii-topk-percent", type=float, default=5.0)
    parser.add_argument("--kii-percentile", type=float, default=95.0)
    parser.add_argument("--smooth-window", type=int, default=7)
    parser.add_argument("--min-duration", type=int, default=5)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--download", action="store_true", help="Run download_data.py if raw data are missing. Requires copernicusmarine login.")
    parser.add_argument("--train-if-missing", action="store_true", help="Train a new model if the checkpoint is missing.")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs used only with --train-if-missing.")
    parser.add_argument("--train-batch-size", type=int, default=16, help="Training batch size used only with --train-if-missing.")
    args = parser.parse_args()

    print("Kuroshio KII one-command runner")
    print("================================")

    if not has_processed_data():
        print("\n[check] Processed data not found.")
        if has_raw_data():
            print("[info] Raw NetCDF files found. Running preprocess.py automatically.")
            run([sys.executable, "preprocess.py"])
        elif args.download:
            print("[info] Raw data not found. Running download_data.py, then preprocess.py.")
            print("[note] This requires a working Copernicus Marine login on this machine.")
            run([sys.executable, "download_data.py"])
            run([sys.executable, "preprocess.py"])
        else:
            print("\n[stop] I cannot find processed data or raw NetCDF files.")
            print("Put your original data/ folder into this project folder, or run:")
            print("    python run_kii.py --download")
            print("after logging in with:")
            print("    copernicusmarine login")
            sys.exit(1)
    else:
        print("[check] Processed data found.")

    if not has_checkpoint(args.checkpoint):
        print(f"\n[check] Checkpoint not found: {args.checkpoint}")
        if args.train_if_missing:
            print("[info] Training a new model automatically.")
            run([
                sys.executable,
                "train.py",
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.train_batch_size),
                "--device",
                args.device,
            ])
        else:
            print("\n[stop] I cannot find the trained model checkpoint.")
            print("Put your original checkpoints/ folder into this project folder, or run:")
            print("    python run_kii.py --train-if-missing")
            sys.exit(1)
    else:
        print(f"[check] Checkpoint found: {args.checkpoint}")

    cmd = [
        sys.executable,
        "evaluate.py",
        "--checkpoint",
        args.checkpoint,
        "--device",
        args.device,
        "--batch_size",
        str(args.batch_size),
        "--score_mode",
        args.score_mode,
        "--topk_percent",
        str(args.topk_percent),
        "--percentile",
        str(args.percentile),
        "--kii_topk_percent",
        str(args.kii_topk_percent),
        "--kii_percentile",
        str(args.kii_percentile),
        "--smooth_window",
        str(args.smooth_window),
        "--min_duration",
        str(args.min_duration),
        "--top_n",
        str(args.top_n),
    ]
    run(cmd)

    print("\nDone. Check these outputs:")
    print("    results/kii_daily_scores.csv")
    print("    results/instability_episodes.csv")
    print("    results/pre_lam_window_stats.csv")
    print("    results/kii_timeseries_2019_2020.png")
    print("    results/kii_zoom_lam_transition.png")


if __name__ == "__main__":
    main()
