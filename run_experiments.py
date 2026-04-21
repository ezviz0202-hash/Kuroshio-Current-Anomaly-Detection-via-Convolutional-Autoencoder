import subprocess
import pandas as pd
from pathlib import Path

CHECKPOINT = "checkpoints/best_model.pt"

# 要尝试的组合（一次性全跑）
configs = [
    {"score_mode": "mean", "topk": 10, "pct": 95},
    {"score_mode": "topk_mean", "topk": 20, "pct": 90},
    {"score_mode": "topk_mean", "topk": 10, "pct": 90},
    {"score_mode": "topk_mean", "topk": 5, "pct": 85},
    {"score_mode": "max", "topk": 1, "pct": 85},
]

results = []

for i, cfg in enumerate(configs):
    print(f"\n===== Running config {i+1}/{len(configs)} =====")
    print(cfg)

    cmd = [
        "python", "evaluate.py",
        "--checkpoint", CHECKPOINT,
        "--device", "cpu",
        "--score_mode", cfg["score_mode"],
        "--topk_percent", str(cfg["topk"]),
        "--percentile", str(cfg["pct"]),
    ]

    subprocess.run(cmd)

    # 读取结果
    df = pd.read_csv("results/anomaly_scores.csv")

    tp = ((df["is_anomaly"] == 1) & (df["date"] >= "2019-08-01")).sum()
    fp = ((df["is_anomaly"] == 1) & (df["date"] < "2019-08-01")).sum()
    fn = ((df["is_anomaly"] == 0) & (df["date"] >= "2019-08-01")).sum()

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    results.append({
        "config": cfg,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

# 保存总结
summary = pd.DataFrame(results)
summary.to_csv("results/experiment_summary.csv", index=False)

print("\n===== FINAL SUMMARY =====")
print(summary.sort_values("f1", ascending=False))