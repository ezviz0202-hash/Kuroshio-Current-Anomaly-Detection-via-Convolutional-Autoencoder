import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from model import KuroshioAutoencoder, masked_mse_loss
DATA_DIR  = Path("data/processed")
CKPT_DIR  = Path("checkpoints")

def load_data(device: torch.device):
    frames      = np.load(DATA_DIR / "kuroshio_frames.npy")   
    land_mask   = np.load(DATA_DIR / "land_mask.npy")          
    splits      = np.load(DATA_DIR / "split_indices.npz")

    train_idx = splits["train"]
    val_idx   = splits["val"]

    X_train = torch.from_numpy(frames[train_idx]).to(device)
    X_val   = torch.from_numpy(frames[val_idx]).to(device)
    mask    = torch.from_numpy(land_mask).to(device)

    return X_train, X_val, mask


def train(args):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == "cpu" else "cpu")
    print(f"Using device: {device}")

    X_train, X_val, mask = load_data(device)

    train_ds = TensorDataset(X_train)
    val_ds   = TensorDataset(X_val)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True,  drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size,
                          shuffle=False, drop_last=False)

    model = KuroshioAutoencoder(in_channels=2,
                                base_filters=args.base_filters).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        model.train()
        train_loss = 0.0
        for (batch,) in train_dl:
            optimizer.zero_grad()
            pred = model(batch)
            loss = masked_mse_loss(pred, batch, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dl)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_dl:
                pred = model(batch)
                val_loss += masked_mse_loss(pred, batch, mask).item()
        val_loss /= len(val_dl)

        scheduler.step(val_loss)
        elapsed = time.time() - t0
        print(f"Epoch {epoch:04d}/{args.epochs}  "
              f"train={train_loss:.5f}  val={val_loss:.5f}  "
              f"({elapsed:.1f}s)")

        log_rows.append([epoch, train_loss, val_loss])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":      epoch,
                "model_state_dict": model.state_dict(),
                "val_loss":   val_loss,
                "args":       vars(args),
            }, CKPT_DIR / "best_model.pt")
            print(f"   Saved best model (val_loss={best_val_loss:.5f})")

    with open(CKPT_DIR / "train_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        writer.writerows(log_rows)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.5f}")
    print(f"Model saved to {CKPT_DIR / 'best_model.pt'}")


def parse_args():
    p = argparse.ArgumentParser(description="Train Kuroshio Autoencoder")
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--base_filters", type=int,   default=32)
    p.add_argument("--device",       type=str,   default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
